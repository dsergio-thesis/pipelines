
import pandas as pd

def label_definitions(classification_labels_file):

    label_definitions = pd.read_csv(classification_labels_file)

    # add column in the beginning, which is an index
    if ("label_index" not in label_definitions.columns):
        label_definitions.insert(0, 'index', range(1, len(label_definitions) + 1))
    label_definitions.to_csv(classification_labels_file, index=False)

    # add index for fast lookup
    label_definitions.set_index('morph_type', inplace=True)

    return label_definitions

