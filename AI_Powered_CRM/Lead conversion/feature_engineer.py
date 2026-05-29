from sklearn.base import BaseEstimator, TransformerMixin

class LeadFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Prevent division by zero errors cleanly
        X["engagement_score"] = (
            X["mail_response_count"] * 2 +
            X["total_meet_count"] * 3 +
            X["website_visits"] * 0.5 +
            X["mail_open_rate"] * 10
        )
        X["budget_per_visit"] = X["budget"] / (X["website_visits"] + 1)
        X["meeting_effectiveness"] = X["mail_response_count"] / (X["total_meet_count"] + 1)
        X["open_visit_ratio"] = X["website_visits"] / (X["mail_open_rate"] + 0.01)
        X["high_intent"] = (
            (X["website_visits"] > 25) & (X["total_meet_count"] >= 2)
        ).astype(int)
        
        # Return ALL columns (raw + engineered) so CatBoost can maximize split choices
        return X