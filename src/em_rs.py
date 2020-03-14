import pandas as pd
from apyori import apriori


def calculate_apriori(transactions, apriori_config):
    results = list(apriori(transactions, **apriori_config))
    rules_list = list()
    for r in results:
        # items_base is the antecedent and the items_add is the consequent.
        antecedent = list(r.ordered_statistics[0].items_base)
        if len(antecedent) == 0:
            continue
        elif len(antecedent) == apriori_config['max_length'] - 1:
            for i in r.ordered_statistics:
                str1 = ','.join(map(str, list(i.items_base)))
                str2 = ','.join(map(str, list(i.items_add)))
                new_tuple = (str1, str2, r.support, i.confidence, i.lift)
                rules_list.append(new_tuple)

    rules = pd.DataFrame(rules_list, columns=['item_A', 'item_B', 'support', 'confidence', 'lift'])

    # check
    rows = rules[(rules['support'] < apriori_config['min_support']) |
                 (rules['confidence'] < apriori_config['min_confidence']) |
                 (rules['lift'] < apriori_config['min_lift'])]
    if rows.shape[0] != 0 or rules.shape[0] == 0:
        raise ValueError("apriori error")

    return rules
