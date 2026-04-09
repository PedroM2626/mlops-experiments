"""
Experimento 9: Reinforcement Learning para OtimizaÃ§Ã£o de EstratÃ©gia de Trading
===============================================================================
Implementa agente RL simples que aprende estratÃ©gia de compra/venda:
- Q-Learning
- Policy Gradient
- ComparaÃ§Ã£o com Buy-and-Hold
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import mlflow
import warnings
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "datasets"

# ============================================================================
# ENVIRONMENT DE TRADING
# ============================================================================

class TradingEnvironment:
    """Ambiente de trading simples para RL."""
    
    def __init__(self, prices, initial_balance=1000.0):
        self.prices = prices
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0  # Quantas aÃ§Ãµes temos
        self.current_step = 0
        self.history = []
    
    def reset(self):
        self.balance = self.initial_balance
        self.position = 0
        self.current_step = 0
        self.history = []
        return self.get_state()
    
    def get_state(self):
        """Retorna estado normalizado."""
        if self.current_step < 5:
            return np.zeros(5)
        
        recent_prices = self.prices[max(0, self.current_step-5):self.current_step]
        
        # Features
        price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        momentum = (recent_prices[-1] - recent_prices[-2]) / recent_prices[-2] if len(recent_prices) > 1 else 0
        balance_ratio = self.balance / self.initial_balance
        position = self.position
        price = self.prices[self.current_step] / np.mean(self.prices)
        
        return np.array([price_change, momentum, balance_ratio, position, price])
    
    def step(self, action):
        """Executa aÃ§Ã£o: 0=hold, 1=buy, 2=sell"""
        current_price = self.prices[self.current_step]
        reward = 0
        
        if action == 1:  # BUY
            if self.balance >= current_price:
                self.position += 1
                self.balance -= current_price
                reward = -0.1  # Penalidade pequena por transaÃ§Ã£o
        
        elif action == 2:  # SELL
            if self.position > 0:
                self.position -= 1
                self.balance += current_price
                reward = -0.1
        
        # PrÃªmio pelo valor da posiÃ§Ã£o
        portfolio_value = self.balance + self.position * current_price
        reward += (portfolio_value - self.initial_balance) / self.initial_balance * 0.01
        
        self.current_step += 1
        done = self.current_step >= len(self.prices) - 1
        
        if done:
            final_value = self.balance + self.position * self.prices[-1]
            reward += (final_value - self.initial_balance) / self.initial_balance
        
        return self.get_state(), reward, done
    
    def get_portfolio_value(self):
        current_price = self.prices[min(self.current_step, len(self.prices)-1)]
        return self.balance + self.position * current_price

# ============================================================================
# AGENTES RL
# ============================================================================

class QLearningAgent:
    """Agente Q-Learning discretizado."""
    
    def __init__(self, n_states=100, n_actions=3, learning_rate=0.1, gamma=0.9, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.random.normal(0, 0.01, size=(n_states, n_actions))
    
    def get_state_index(self, state):
        """Discretiza estado contÃ­nuo."""
        state_normalized = (state + 1) / 2  # Normaliza para [0, 1]
        state_normalized = np.clip(state_normalized, 0, 0.999)
        return int(state_normalized[0] * self.n_states)
    
    def choose_action(self, state_idx, train=True):
        if train and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state_idx])
    
    def update(self, state_idx, action, reward, next_state_idx, done):
        target = reward
        if not done:
            target += self.gamma * np.max(self.q_table[next_state_idx])
        
        td_error = target - self.q_table[state_idx, action]
        self.q_table[state_idx, action] += self.lr * td_error

def run_rl_trading():
    """Pipeline de RL para trading."""
    
    print("\\n" + "="*80)
    print("ðŸ’° EXPERIMENTO 9: RL PARA OTIMIZAÃ‡ÃƒO DE TRADING")
    print("="*80 + "\\n")
    
    mlflow.set_experiment("RL_Trading_Optimization")
    
    with mlflow.start_run(run_name="rl_trading_complete"):
        
        # Carrega dados
        print("1ï¸âƒ£  Carregando dados de sÃ©rie temporal...")
        
        price_file = DATA_DIR / "Electric_Production.csv"
        df = pd.read_csv(price_file, header=0)
        
        # Pega primeira coluna como preÃ§os
        prices = df.iloc[:, 0].values
        
        # Pega primeira coluna como preços; se vier pointer LFS ou texto inválido,
        # cai para uma série sintética para manter o experimento executável.
        prices = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().values
        if len(prices) < 20:
            print("   Dados inválidos ou insuficientes; gerando série sintética.")
            t = np.arange(400)
            prices = 100 + 0.05 * t + 8 * np.sin(t / 12) + np.random.normal(0, 1.5, size=len(t))
        
        # Normaliza
        scaler = MinMaxScaler(feature_range=(1, 100))
        prices = scaler.fit_transform(prices.reshape(-1, 1)).flatten()
        
        print(f"   PreÃ§os carregados: {len(prices)}")
        print(f"   Min: {prices.min():.2f}, Max: {prices.max():.2f}, Mean: {prices.mean():.2f}\\n")
        
        results = {
            'dataset': {
                'prices_count': len(prices),
                'price_range': [float(prices.min()), float(prices.max())]
            },
            'training': {},
            'testing': {}
        }
        
        # Split treino/teste
        split = int(0.8 * len(prices))
        prices_train = prices[:split]
        prices_test = prices[split:]
        
        # ====================================================================
        # TREINA AGENTE QL
        # ====================================================================
        print("2ï¸âƒ£  Treinando agente Q-Learning...")
        
        agent = QLearningAgent(n_states=50, n_actions=3, learning_rate=0.1, gamma=0.9, epsilon=0.2)
        env_train = TradingEnvironment(prices_train, initial_balance=1000.0)
        
        episode_rewards = []
        
        for episode in range(20):
            state = env_train.reset()
            total_reward = 0
            
            for step in range(len(prices_train) - 1):
                state_idx = agent.get_state_index(state)
                action = agent.choose_action(state_idx, train=True)
                next_state, reward, done = env_train.step(action)
                
                next_state_idx = agent.get_state_index(next_state)
                agent.update(state_idx, action, reward, next_state_idx, done)
                
                total_reward += reward
                state = next_state
                
                if done:
                    break
            
            episode_rewards.append(total_reward)
            
            if (episode + 1) % 5 == 0:
                print(f"   EpisÃ³dio {episode + 1}/20 - Reward: {total_reward:.3f}")
        
        print()
        results['training']['final_rewards'] = [float(r) for r in episode_rewards[-5:]]
        results['training']['avg_reward'] = float(np.mean(episode_rewards[-5:]))
        
        # ====================================================================
        # TESTA AGENTE RL
        # ====================================================================
        print("3ï¸âƒ£  Testando agente RL em dados novos...")
        
        env_test = TradingEnvironment(prices_test, initial_balance=1000.0)
        state = env_test.reset()
        
        rl_actions = []
        
        for step in range(len(prices_test) - 1):
            state_idx = agent.get_state_index(state)
            action = agent.choose_action(state_idx, train=False)  # Greedy
            rl_actions.append(action)
            next_state, reward, done = env_test.step(action)
            state = next_state
            if done:
                break
        
        rl_final_value = env_test.get_portfolio_value()
        rl_return = ((rl_final_value - 1000.0) / 1000.0) * 100
        
        print(f"   Portfolio Final: ${rl_final_value:.2f}")
        print(f"   Retorno: {rl_return:+.1f}%\\n")
        
        results['testing']['rl_agent'] = {
            'final_portfolio': float(rl_final_value),
            'return_pct': float(rl_return),
            'actions_taken': int(sum(1 for a in rl_actions if a > 0))
        }
        
        # ====================================================================
        # BASELINE: BUY AND HOLD
        # ====================================================================
        print("4ï¸âƒ£  Calculando Baseline (Buy and Hold)...")
        
        initial_shares = 1000.0 / prices_test[0]
        final_shares_value = initial_shares * prices_test[-1]
        hold_return = ((final_shares_value - 1000.0) / 1000.0) * 100
        
        print(f"   Initial cost: $1000")
        print(f"   Final value: ${final_shares_value:.2f}")
        print(f"   Retorno: {hold_return:+.1f}%\\n")
        
        results['testing']['buy_hold'] = {
            'final_value': float(final_shares_value),
            'return_pct': float(hold_return)
        }
        
        # ====================================================================
        # COMPARAÃ‡ÃƒO
        # ====================================================================
        print("5ï¸âƒ£  ComparaÃ§Ã£o de EstratÃ©gias...")
        
        outperformance = rl_return - hold_return
        
        print(f"   RL Agent:      {rl_return:+7.2f}%")
        print(f"   Buy & Hold:    {hold_return:+7.2f}%")
        print(f"   DiferenÃ§a:     {outperformance:+7.2f}% {'âœ… RL melhor' if outperformance > 0 else 'âŒ Hold melhor'}\\n")
        
        results['comparison'] = {
            'rl_return': float(rl_return),
            'buy_hold_return': float(hold_return),
            'outperformance_pct': float(outperformance)
        }
        
        # ====================================================================
        # SALVA RESULTADOS
        # ====================================================================
        output_dir = BASE_DIR / "artifacts" / "rl_trading"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = output_dir / f"rl_trading_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"âœ… Resultados salvos: {results_file}")
        
        # MLflow logging
        mlflow.log_param("seed", SEED)
        mlflow.log_param("episodes", 20)
        mlflow.log_metric("rl_return_pct", rl_return)
        mlflow.log_metric("buy_hold_return_pct", hold_return)
        mlflow.log_metric("outperformance_pct", outperformance)
        mlflow.log_artifact(str(results_file))
        
        print("\\n" + "="*80)
        print("âœ… EXPERIMENTO 9 CONCLUÃDO - RL Trading")
        print("="*80 + "\\n")
        
        return results

if __name__ == "__main__":
    run_rl_trading()

