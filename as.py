import numpy as np

def extract_loo_masked_features(strings, alpha=0.1):
    """
    n개의 5자리 숫자 문자열 리스트에서 Leave-One-Out(LOO) 방식의 
    '한 자리 가리고 맞히기' 통계 피처를 추출합니다.
    
    Parameters:
    - strings: 길이 5인 숫자 문자열들의 리스트 (예: ["23581", "16274", ...]), 크기 n (1 <= n <= 100)
    - alpha: 라플라스 스무딩 파라미터 (한 번도 안 나온 숫자에 부여할 기본 가중치)
    
    Returns:
    - features: (5, 7) 크기의 NumPy 배열 (각 자릿수별 7개의 예측성 피처)
    """
    # 1. 문자열 리스트를 (n, 5) 크기의 정수 배열(0~9)로 변환
    X = np.array([[int(c) for c in s] for s in strings], dtype=np.int32)
    n, D = X.shape  # n: 샘플 수, D: 5 (자릿수)
    
    # 예외 처리: n=1일 때는 나머지 n-1개 샘플이 없으므로 균등 분포(완전 무작위) 기준값 반환
    if n <= 1:
        uniform_loss = np.log(10.0) # 약 2.302585
        default_feat = np.array([
            uniform_loss, uniform_loss, 0.1, 0.1, uniform_loss, uniform_loss, 0.0
        ], dtype=np.float32)
        return np.tile(default_feat, (D, 1))
    
    # 2. (n, 5, 10) 크기의 One-Hot 인코딩 텐서 생성
    X_oh = np.eye(10, dtype=np.float32)[X]
    
    # 각 자리 i별 7가지 핵심 피처를 담을 배열
    # [0]: best_pair_loss (나를 가장 잘 맞히는 단 하나의 짝꿍 자릿수 기준 Loss -> 다중 Fake 대비 핵심!)
    # [1]: mean_pair_loss (나머지 4개 자릿수 전체 평균 기준 Loss)
    # [2]: best_pair_acc  (가장 잘 맞히는 짝꿍 자릿수의 Top-1 정답률)
    # [3]: ensemble_acc   (4개 자릿수 앙상블 예측의 Top-1 정답률)
    # [4]: ensemble_loss  (4개 자릿수 가중 앙상블 예측의 Loss)
    # [5]: pred_entropy   (예측 확률 분포의 평균 엔트로피 - 찍기 여부 판별)
    # [6]: self_loo_loss  (다른 자릿수 안 보고 '내 자릿수의 나머지 n-1개 분포'만 보고 찍었을 때의 Loss)
    features = np.zeros((D, 7), dtype=np.float32)
    row_idx = np.arange(n)
    
    for i in range(D):
        # ---------------------------------------------------------
        # A. 단독 분포 기반 LOO 예측 (Baseline: 예시 4번 '11112' 케이스 검출용)
        # ---------------------------------------------------------
        # 전체 합에서 내 자신(k번째 샘플)의 One-Hot을 빼면 나머지 n-1개의 빈도가 됨! (for문 불필요)
        self_counts_loo = X_oh[:, i, :].sum(axis=0, keepdims=True) - X_oh[:, i, :] # (n, 10)
        self_probs_loo = (self_counts_loo + alpha) / ((n - 1) + 10.0 * alpha)      # (n, 10)
        
        # 실제 가려진 숫자 X[:, i]를 맞혔을 때의 확률 및 Loss
        true_self_probs = self_probs_loo[row_idx, X[:, i]]
        self_loo_loss = -np.mean(np.log(true_self_probs + 1e-12))
        
        # ---------------------------------------------------------
        # B. 다른 자릿수 j를 보고 자릿수 i를 맞히는 조건부 LOO 예측
        # ---------------------------------------------------------
        pair_losses = []
        pair_accs = []
        pair_probs_list = []
        
        for j in range(D):
            if i == j:
                continue
                
            # 전체 샘플에서 (자릿수 j의 숫자, 자릿수 i의 숫자) 동시 등장 빈도 행렬 (10, 10)
            # C_ji[a, b] = j자리에 a가 오고 i자리에 b가 온 횟수
            C_ji = X_oh[:, j, :].T @ X_oh[:, i, :]  # (10, 10)
            
            # 각 샘플 k의 j번째 자릿수 숫자(X[k, j])에 해당하는 행만 추출 -> (n, 10)
            counts_given_j = C_ji[X[:, j], :]
            
            # [핵심] Leave-One-Out: 현재 샘플 k 자신의 등장 횟수(1)를 빼줌!
            # 이렇게 하면 k번째 샘플을 빼고 나머지 n-1개로만 만든 조건부 빈도표가 즉시 완성됨
            counts_loo = counts_given_j - X_oh[:, i, :]  # (n, 10)
            
            # 라플라스 스무딩을 적용해 조건부 확률 P(X_i | X_j = x_{k,j}) 계산
            denom = counts_loo.sum(axis=1, keepdims=True) + 10.0 * alpha
            probs_loo = (counts_loo + alpha) / denom  # (n, 10)
            
            # 1) 실제 가려진 숫자(X[:, i])에 대한 예측 확률 및 NLL(Cross-Entropy Loss)
            true_probs = probs_loo[row_idx, X[:, i]]
            loss_j = -np.mean(np.log(true_probs + 1e-12))
            
            # 2) 가장 확률이 높다고 찍은 숫자가 실제 숫자와 일치하는지 (Top-1 정답률)
            pred_digits = np.argmax(probs_loo, axis=1)
            acc_j = np.mean(pred_digits == X[:, i])
            
            pair_losses.append(loss_j)
            pair_accs.append(acc_j)
            pair_probs_list.append(probs_loo)
            
        pair_losses = np.array(pair_losses)
        pair_accs = np.array(pair_accs)
        
        # ---------------------------------------------------------
        # C. 4개 자릿수의 예측치를 결합한 앙상블(Ensemble) 예측
        # ---------------------------------------------------------
        # Fake 자릿수가 섞여 있을 때 단순 평균을 내면 노이즈가 섞이므로,
        # Loss가 낮았던(신뢰도가 높은) 자릿수 j에 더 큰 가중치를 주어 확률 분포를 합침 (Softmax 가중합)
        weights = np.exp(-pair_losses * 2.0)
        weights /= np.sum(weights)
        
        ensemble_probs = np.zeros((n, 10), dtype=np.float32)
        for w, p_loo in zip(weights, pair_probs_list):
            ensemble_probs += w * p_loo
            
        # 앙상블 예측의 Loss, 정답률, 예측 엔트로피(불확실성) 계산
        ens_true_probs = ensemble_probs[row_idx, X[:, i]]
        ensemble_loss = -np.mean(np.log(ens_true_probs + 1e-12))
        ensemble_acc = np.mean(np.argmax(ensemble_probs, axis=1) == X[:, i])
        
        # 예측 분포 자체의 샤논 엔트로피: 확신을 갖고 찍으면 0에 가깝고, 찍기(Fake)면 2.302에 가까움
        pred_entropy = -np.mean(np.sum(ensemble_probs * np.log(ensemble_probs + 1e-12), axis=1))
        
        # 자릿수 i의 최종 피처 저장
        features[i] = [
            np.min(pair_losses),   # 0: 나를 가장 잘 설명해주는 단짝 자릿수와의 Loss (낮을수록 Real)
            np.mean(pair_losses),  # 1: 나머지 4개 자릿수 전체 평균 Loss
            np.max(pair_accs),     # 2: 단짝 자릿수로 맞혔을 때의 최대 정답률 (높을수록 Real)
            ensemble_acc,          # 3: 4개 자릿수 종합 정답률
            ensemble_loss,         # 4: 4개 자릿수 종합 Loss
            pred_entropy,          # 5: 예측 분포의 불확실성(엔트로피)
            self_loo_loss          # 6: 나 자신의 단독 분포 LOO Loss ('11112' 같은 상수열 잡기용)
        ]
        
    return features
