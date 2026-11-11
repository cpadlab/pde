
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


class AssociationRuleScorer:
    """
    Extracts association rules and scores observations that violate learned logic.

    Mines frequent itemsets using the Apriori algorithm and generates association
    rules filtered by minimum support, confidence, and lift thresholds. Scores
    new transactional records based on rule violations (antecedent present but
    consequent absent), weighting penalties by lift and providing human-readable
    diagnostic explanations.

    Attributes:
        min_support (float): Minimum support threshold required for frequent
            itemset mining.
        min_confidence (float): Minimum confidence threshold required to keep an
            association rule.
        min_lift (float): Minimum lift threshold required to keep an association
            rule.
        max_len (int): Maximum length of itemsets evaluated during Apriori.
        rules_df_ (Optional[pd.DataFrame]): DataFrame containing mined and filtered
            association rules with metadata columns (`antecedents_str`,
            `consequents_str`, `rule_repr`). None before `fit`.
        is_fitted_ (bool): Flag indicating whether association rules have been
            mined.
    """

    def __init__(
        self, min_support: float = 0.05, min_confidence: float = 0.50,
        min_lift: float = 1.10, max_len: int = 3
    ) -> None:
        """
        Initializes the AssociationRuleScorer with rule-mining hyperparameters.

        Args:
            min_support (float, optional): Minimum support threshold for Apriori.
                Defaults to 0.05.
            min_confidence (float, optional): Minimum confidence threshold for
                association rules. Defaults to 0.50.
            min_lift (float, optional): Minimum lift threshold for association
                rules. Defaults to 1.10.
            max_len (int, optional): Maximum length of itemsets evaluated by
                Apriori. Defaults to 3.
        """
        self.min_support = float(min_support)
        self.min_confidence = float(min_confidence)
        self.min_lift = float(min_lift)
        self.max_len = int(max_len)

        self.rules_df_: Optional[pd.DataFrame] = None
        self.is_fitted_: bool = False


    def fit(self, transactions_df: pd.DataFrame) -> "AssociationRuleScorer":
        """
        Learns frequent co-occurrence rules from a one-hot encoded dataset.

        Executes the Apriori algorithm and extracts association rules that meet
        the specified support, confidence, and lift thresholds.

        Args:
            transactions_df (pd.DataFrame): One-hot encoded boolean DataFrame of
                transactions/observations where rows correspond to samples and
                columns to categorical items.

        Returns:
            AssociationRuleScorer: The fitted scorer instance (`self`).
        """
        frequent = apriori(
            transactions_df,
            min_support=self.min_support,
            use_colnames=True,
            max_len=self.max_len,
        )

        if frequent.empty:
            self.rules_df_ = pd.DataFrame()
            self.is_fitted_ = True

            return self

        rules = association_rules(
            frequent,
            metric="confidence",
            min_threshold=self.min_confidence,
        )

        if rules.empty:
            self.rules_df_ = pd.DataFrame()
            self.is_fitted_ = True

            return self

        rules = rules[rules["lift"] >= self.min_lift].copy()
        rules["antecedents_str"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(list(s))))
        rules["consequents_str"] = rules["consequents"].apply(lambda s: ", ".join(sorted(list(s))))
        rules["rule_repr"] = rules["antecedents_str"] + " => " + rules["consequents_str"]

        self.rules_df_ = rules
        self.is_fitted_ = True

        return self


    def score(self, transactions_df: pd.DataFrame) -> Tuple[np.ndarray, List[List[str]]]:
        """
        Calculates lift-weighted violation scores and diagnostic explanations.

        A violation occurs when a record satisfies all items in a rule's
        antecedent but fails to satisfy all items in its consequent. The
        resulting score is the sum of violated rule lifts normalized by the
        total applicable lift weight for that sample.

        Args:
            transactions_df (pd.DataFrame): One-hot encoded boolean DataFrame
                of observations to score.

        Returns:
            Tuple[np.ndarray, List[List[str]]]: A tuple containing:
                - **scores** (`np.ndarray`): 1D array of float violation scores,
                  where higher values indicate stronger deviation from expected
                  co-occurrence patterns.
                - **explanations** (`List[List[str]]`): List containing a list
                  of violated rule string representations (`'rule_repr'`) for
                  each observation.

        Raises:
            RuntimeError: If called before fitting rules via `fit`.
        """
        if not self.is_fitted_:
            raise RuntimeError("You must call .fit() before scoring observations.")

        if self.rules_df_ is None or self.rules_df_.empty:
            return np.zeros(len(transactions_df)), [[] for _ in range(len(transactions_df))]

        all_cols = set(transactions_df.columns)
        antecedent_masks = []
        consequent_masks = []

        lifts = self.rules_df_["lift"].to_numpy()
        rule_texts = self.rules_df_["rule_repr"].tolist()

        for _, r in self.rules_df_.iterrows():
            ant_cols = [c for c in list(r["antecedents"]) if c in all_cols]
            con_cols = [c for c in list(r["consequents"]) if c in all_cols]

            ant_m = transactions_df[ant_cols].all(axis=1).to_numpy() if ant_cols else np.zeros(len(transactions_df), dtype=bool)
            con_m = transactions_df[con_cols].all(axis=1).to_numpy() if con_cols else np.zeros(len(transactions_df), dtype=bool)

            antecedent_masks.append(ant_m)
            consequent_masks.append(con_m)

        A = np.column_stack(antecedent_masks)
        C = np.column_stack(consequent_masks)
        violations = A & (~C)

        applicable_weights = np.maximum(1.0, np.sum(A * lifts, axis=1))
        scores = np.sum(violations * lifts, axis=1) / applicable_weights

        explanations = []
        for i in range(len(transactions_df)):
            broken = np.nonzero(violations[i])[0]
            explanations.append([rule_texts[idx] for idx in broken])

        return scores, explanations