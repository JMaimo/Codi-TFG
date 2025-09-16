import os, csv, sys
from dataclasses import dataclass
from typing import Tuple, List
import numpy as np

# =========================
# Paràmetres del joc i constants de recompensa
# =========================
LIE_PROB   = 1.0/3.0
TRUTH_PROB = 1.0 - LIE_PROB

# =========================
# Algoritmes i utilitats
# =========================
@dataclass
class EGreedyParams:
    epsilon: float = 0.1
    alpha: float = 0.1

@dataclass
class GBParams:
    alpha: float = 0.1   # taxa d'aprenentatge

def get_reward(player_card: int, real_card: int) -> int:
    """Recompensa: negativa si la suma és imparell, positiva si és parell"""
    total = player_card + real_card
    return total if total % 2 == 1 else -total

def dealer_draw(pi: float, rng: np.random.RandomState) -> int:
    """Carta real del crupier: 1 amb prob pi, sinó 2."""
    return 1 if rng.rand() < pi else 2

def dealer_says(real_card: int, truth_prob: float, rng: np.random.RandomState) -> int:
    """El crupier pot mentir."""
    if rng.rand() < truth_prob:
        return real_card
    return 1 if real_card == 2 else 2

def egreeedy_choice(q1: float, q2: float, epsilon: float, rng: np.random.RandomState) -> int:
    if rng.rand() < epsilon:
        return 1 if rng.rand() < 0.5 else 2
    if abs(q1 - q2) < 1e-12:
        return 1 if rng.rand() < 0.5 else 2
    return 1 if q1 > q2 else 2

def softmax(z):
    e = np.exp(z)
    return e / e.sum()

# =========================
# Entrenament
# =========================
def train_epsilon_greedy(N_train: int, pi: float, truth_prob: float, params: EGreedyParams, rng: np.random.RandomState):
    """
    Entrenament de l'ε-greedy: actualitza Q durant N_train passos.
    Retorna els valors Q finals.
    """
    Q = {'diu1': {1: 0.0, 2: 0.0}, 'diu2': {1: 0.0, 2: 0.0}}
    avg_reward = 0.0

    for t in range(1, N_train + 1):
        real = dealer_draw(pi, rng)
        said = dealer_says(real, truth_prob, rng)
        state = 'diu1' if said == 1 else 'diu2'

        a = egreeedy_choice(Q[state][1], Q[state][2], params.epsilon, rng)
        r = get_reward(a, real)

        # Actualitza Q
        Q[state][a] += params.alpha * (r - Q[state][a])

        # Baseline opcional (per seguir mètriques)
        avg_reward += (r - avg_reward) / t

    return Q

def train_gradient_bandit(N_train: int, pi: float, alpha: float, rng: np.random.RandomState):
    """
    Algorisme Gradient Bandit amb baseline (equació 3.10 de Sutton & Barto).
    Entrenament durant N_train passos i retorn de les preferències finals.
    """
    H = np.zeros((2, 2))   # preferències H[estat, acció]
    avg_reward = 0.0       # baseline

    for t in range(1, N_train + 1):
        # 1) Crupier treu carta i pot mentir
        d_real = dealer_draw(pi, rng)
        d_say = dealer_says(d_real, truth_prob=2/3, rng=rng)
        s_idx = d_say - 1

        # 2) Política softmax sobre H
        pi_probs = softmax(H[s_idx])
        a_idx = rng.choice([0, 1], p=pi_probs)
        a_player = a_idx + 1

        # 3) Obtenir recompensa
        r = get_reward(a_player, d_real)

        # 4) Actualitzar baseline
        avg_reward += (r - avg_reward) / t

        # 5) Actualitzar preferències H (gradient ascent)
        for a in range(2):
            if a == a_idx:
                H[s_idx, a] += alpha * (r - avg_reward) * (1 - pi_probs[a])
            else:
                H[s_idx, a] -= alpha * (r - avg_reward) * pi_probs[a]

    # política final després d’entrenar
    probs_final = [softmax(H[0]), softmax(H[1])]
    return H, probs_final


# =========================
# Joc
# =========================

def run_episode_EG(Q, N_play: int, pi: float, truth_prob: float, params: EGreedyParams, rng: np.random.RandomState):
    """
    Joc amb la política entrenada (Q fixos).
    No s'actualitzen els Q.
    """
    rewards = []
    for _ in range(N_play):
        real = dealer_draw(pi, rng)
        said = dealer_says(real, truth_prob, rng)
        state = 'diu1' if said == 1 else 'diu2'

        # Selecciona acció segons Q fixos
        a = egreeedy_choice(Q[state][1], Q[state][2], params.epsilon, rng)
        r = get_reward(a, real)
        rewards.append(r)
    return rewards

def run_episode_GB(probs_final, N_play: int, pi, rng):
    rewards = []
    for _ in range(N_play):
        d_real = dealer_draw(pi, rng)
        d_say = dealer_says(d_real, truth_prob=2/3, rng=rng)
        context = d_say - 1

        action_idx = rng.choice([0, 1], p=probs_final[context])
        a_player = action_idx + 1

        r = get_reward(a_player, d_real)
        rewards.append(r)
    return rewards

# =========================
# Escriptura i utilitats
# =========================
def frac_str(num: int, den: int) -> str:
    return f"{num}-{den}"

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def write_lines(out_path: str, algorithm: str, pi: float, N: int, rep: int, rewards: List[int]) -> None:
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f, delimiter=';')
        for i, r in enumerate(rewards, start=1):
            w.writerow([algorithm, f"{pi:.12g}", N, rep, i, r])

def progress_bar(current: int, total: int, length: int = 40):
    filled = int(length * current // total)
    bar = "█" * filled + "-" * (length - filled)
    sys.stdout.write(f"\r|{bar}| {current}/{total}")
    sys.stdout.flush()

# =========================
# Bucle principal
# =========================
def generate_all(
    X: int = 100,
    Ns: List[int] = [100, 500, 1000, 10000],
    pis: List[Tuple[int, int]] = [(7, 22), (21, 26), (6, 12), (1, 3), (2, 3), (1, 4), (3, 4), (5, 12), (7, 12), (9, 10)],
    eg_params: EGreedyParams = EGreedyParams(epsilon=0.1, alpha=0.1),
    bz_params: GBParams = GBParams(alpha=0.1),
    out_dir: str = "resultats",
    base_seed: int = 12345,
    truth_prob: float = TRUTH_PROB
):
    ensure_dir(out_dir)
    total = 2 * len(pis) * len(Ns) * X
    done = 0

    for (num, den) in pis:
        pi = num / den
        pi_token = frac_str(num, den)
        for N in Ns:
            for rep in range(1, X + 1):
                seed = (base_seed * 1_000_003 + hash((num, den, N, rep))) & 0xFFFFFFFF
                rng_E = np.random.RandomState(seed)
                rng_B = np.random.RandomState((seed + 17) & 0xFFFFFFFF)

                # --- ε-greedy ---
                Q_final = train_epsilon_greedy(N, pi, truth_prob, eg_params, rng_E)
                rewards_EG = run_episode_EG(Q_final, N, pi, truth_prob, eg_params, rng_E)
                file_EG = os.path.join(out_dir, f"EGREEDY_{pi_token}_{N}_{rep}.csv")
                write_lines(file_EG, "EGREEDY", pi, N, rep, rewards_EG)
                done += 1;
                progress_bar(done, total)

                # --- Gradient Bandit (Boltzmann) ---
                Q, probs_final = train_gradient_bandit(N, pi, bz_params.alpha, rng_B)
                rewards_BZ = run_episode_GB(probs_final, N, pi, rng_B)
                file_BZ = os.path.join(out_dir, f"BOLTZMANN_{pi_token}_{N}_{rep}.csv")
                write_lines(file_BZ, "BOLTZMANN", pi, N, rep, rewards_BZ)
                done += 1; progress_bar(done, total)

    print("\nFi! Fitxers generats a:", out_dir)

# =========================
# Main
# =========================
if __name__ == "__main__":
    generate_all(
        X=100,
        Ns=[100, 500, 1000, 10000],
        pis=[(7, 17), (14, 19), (6, 12), (1, 3), (2, 3), (1, 4), (3, 4), (5, 12), (7, 12), (9, 10)],
        eg_params=EGreedyParams(epsilon=0.1, alpha=0.1),
        bz_params=GBParams(alpha=0.1),
        out_dir="resultats",
        base_seed=12345,
        truth_prob=2 / 3,
    )
