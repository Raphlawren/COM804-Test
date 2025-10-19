from typing import Tuple
import pandas as pd
import numpy as np

class CBR:
    def __init__(self, continuous_features: list, categorical_features: list, target_names: dict, 
                 feature_weights: dict = None, normalize_method: str = 'minmax'):
        # Initialize CBR with feature types, labels, weights, and normalization
        self.case_base = []  # store all past cases
        self.features_names = continuous_features + categorical_features
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        self.target_names = target_names
        self.feature_weights = feature_weights or {f: 1.0 for f in self.features_names}
        self.normalize_method = normalize_method
        self.feature_ranges = {}  

    def create_case_base(self, X: pd.DataFrame, y: pd.Series) -> list:
        # Build the case base by combining features and target into structured cases
        # calculate min, max, mean, std for continuous features
        self._compute_feature_ranges(X)  
        for i, (features, target) in enumerate(zip(X.values, y.values)):
            case = {
                'id': i,
                'features': features.tolist(),
                'solution': int(target),
                'solution_name': self.target_names[int(target)]
            }
            self.case_base.append(case)
    
    def _compute_feature_ranges(self, X: pd.DataFrame):
        # Compute stats used for similarity normalization of continuous features
        for feature in self.features_names:
            if feature in self.continuous_features:
                col_values = X[feature].dropna()
                self.feature_ranges[feature] = {
                    'min': col_values.min(),
                    'max': col_values.max(),
                    'std': col_values.std(),
                    'mean': col_values.mean()
                }

    def local_similarity(self, case1: dict, case2: dict) -> float:
        # Compute weighted similarity between two cases feature-by-feature
        sim = 0.0
        total_weight = 0.0
        
        for i, feature in enumerate(self.features_names):
            weight = self.feature_weights.get(feature, 1.0)
            val1, val2 = case1['features'][i], case2['features'][i]
            
            if pd.isna(val1) or pd.isna(val2):
                feature_sim = 0.5  # neutral similarity if missing
            elif feature in self.continuous_features:
                # normalize continuous difference and convert to similarity
                stats = self.feature_ranges.get(feature, {})
                min_val, max_val = stats.get('min', 0), stats.get('max', 1)
                range_val = max_val - min_val
                
                if range_val == 0:
                    feature_sim = 1.0 if val1 == val2 else 0.0
                else:
                    if self.normalize_method == 'minmax':
                        feature_sim = 1 - abs(val1 - val2) / range_val
                    elif self.normalize_method == 'gaussian':
                        std_val = stats.get('std', 1)
                        feature_sim = np.exp(-(abs(val1 - val2) ** 2) / (2 * std_val ** 2)) if std_val > 0 else float(val1 == val2)
                    else:
                        feature_sim = 1 - abs(val1 - val2) / range_val
                        
            elif feature in self.categorical_features:
                feature_sim = 1.0 if val1 == val2 else 0.0  # 1 if same category
            
            sim += feature_sim * weight
            total_weight += weight
        
        return sim / total_weight if total_weight > 0 else 0.0

    def global_similarity(self, new_case: dict) -> list:
        # Compute similarity of a new case with all cases in the base
        similarities = [(case, self.local_similarity(new_case, case)) for case in self.case_base]
        similarities.sort(key=lambda x: x[1], reverse=True)  # sort by similarity
        return similarities

    def retrieve_similar_cases(self, query_case: dict, k: int = 5, exclude_self: bool = False) -> list:
        # Retrieve top-k most similar cases for a query
        similarities = self.global_similarity(query_case)
        if exclude_self:
            query_id = query_case.get('id')
            similarities = [(case, sim) for case, sim in similarities if case.get('id') != query_id]
        return similarities[:k]
    
    def predict_proba(self, X: pd.DataFrame, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        # Compute class probabilities from top-k neighbors
        assert self._y_cases is not None
        classes = np.unique(self._y_cases)
        probs = np.zeros((len(X), len(classes)), dtype=float)
        for i in range(len(X)):
            sims = self._similarity_to_all(X.iloc[i][self.all_cols].values)
            top_idx = np.argsort(-sims)[:max(k, 1)]
            labels = self._y_cases[top_idx]
            vals, counts = np.unique(labels, return_counts=True)
            probs[i, np.searchsorted(classes, vals)] = counts / max(k, 1)
        return probs, classes

    def reuse_solution(self, similar_cases: list, voting_method: str = 'majority') -> int:
        # Aggregate solutions from retrieved cases to predict label
        if not similar_cases:
            return None
        votes = {}
        if voting_method == 'weighted':
            total_sim = sum(sim for _, sim in similar_cases)
            for case, sim in similar_cases:
                votes[case['solution']] = votes.get(case['solution'], 0) + (sim / total_sim if total_sim > 0 else sim)
        elif voting_method == 'distance_weighted':
            for case, sim in similar_cases:
                votes[case['solution']] = votes.get(case['solution'], 0) + sim ** 2
        elif voting_method == 'majority':
            for case, _ in similar_cases:
                votes[case['solution']] = votes.get(case['solution'], 0) + 1
        else:
            for case, sim in similar_cases:
                votes[case['solution']] = votes.get(case['solution'], 0) + sim
        return max(votes, key=votes.get)  # return label with highest vote
    
    def optimize_weights_gradient_descent(self, X_val: pd.DataFrame, y_val: pd.Series, 
                                          learning_rate: float = 0.01, epochs: int = 100, 
                                          k: int = 5, verbose: bool = True):
        # Optimize feature weights using gradient descent on validation data
        n_features = len(self.features_names)
        weights = np.array([self.feature_weights.get(f, 1.0) for f in self.features_names])
        history = {'epoch': [], 'accuracy': [], 'loss': []}
        best_weights, best_accuracy = weights.copy(), 0.0
        
        for epoch in range(epochs):
            gradients = np.zeros(n_features)
            total_loss = 0.0
            correct_predictions = 0
            
            for idx in range(len(X_val)):
                query_features = X_val.iloc[idx].values
                true_label = int(y_val.iloc[idx])
                query_case = {'id': idx, 'features': query_features.tolist()}
                
                similar_cases = self.retrieve_similar_cases(query_case, k=k, exclude_self=True)
                predicted_label = self.reuse_solution(similar_cases, voting_method='weighted')
                
                if predicted_label == true_label:
                    correct_predictions += 1
                
                # Compute error gradient for each feature
                for case, similarity in similar_cases:
                    target_sim = 1.0 if case['solution'] == true_label else 0.0
                    error = similarity - target_sim
                    total_loss += error ** 2
                    
                    for i, feature in enumerate(self.features_names):
                        val_query = query_features[i]
                        val_case = case['features'][i]
                        if pd.isna(val_query) or pd.isna(val_case):
                            continue
                        
                        if feature in self.continuous_features:
                            stats = self.feature_ranges.get(feature, {})
                            min_val, max_val = stats.get('min', 0), stats.get('max', 1)
                            range_val = max_val - min_val
                            if range_val > 0:
                                if self.normalize_method == 'minmax':
                                    feature_sim = 1 - abs(val_query - val_case) / range_val
                                else:
                                    std_val = stats.get('std', 1)
                                    feature_sim = np.exp(-(abs(val_query - val_case) ** 2) / (2 * std_val ** 2)) if std_val > 0 else float(val_query == val_case)
                                total_weight = sum(weights)
                                d_weight = (feature_sim / total_weight) - (similarity * weights[i] / (total_weight ** 2))
                                gradients[i] += 2 * error * d_weight
                        
                        elif feature in self.categorical_features:
                            feature_sim = 1.0 if val_query == val_case else 0.0
                            total_weight = sum(weights)
                            d_weight = (feature_sim / total_weight) - (similarity * weights[i] / (total_weight ** 2))
                            gradients[i] += 2 * error * d_weight
            
            # Update weights using gradients
            gradients /= len(X_val)
            weights -= learning_rate * gradients
            weights = np.maximum(weights, 0.01)  # avoid zero weights
            weights = weights / np.sum(weights) * n_features  # normalize weights
            
            # Update class weights dictionary
            for i, feature in enumerate(self.features_names):
                self.feature_weights[feature] = weights[i]
            
            accuracy = correct_predictions / len(X_val)
            avg_loss = total_loss / len(X_val)
            history['epoch'].append(epoch)
            history['accuracy'].append(accuracy)
            history['loss'].append(avg_loss)
            
            # Track best weights
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weights = weights.copy()
            
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch:3d}/{epochs}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")
        
        # Set final weights to best found
        for i, feature in enumerate(self.features_names):
            self.feature_weights[feature] = best_weights[i]
        
        if verbose:
            print(f"\nOptimization complete!")
            print(f"Best accuracy: {best_accuracy:.4f}")
            sorted_weights = sorted(self.feature_weights.items(), key=lambda x: x[1], reverse=True)
            for feature, weight in sorted_weights:
                print(f"  {feature:30s}: {weight:.4f}")
        
        return history
