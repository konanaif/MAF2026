"""Restore upstream Adult's four binary sensitive attributes without changing row IDs."""
from pathlib import Path
import numpy as np
import pandas as pd


def extend_adult_sensitive(dataset, data_directory):
    directory = Path(data_directory)
    options = dict(header=None, names=list(range(15)), usecols=[0, 5, 8, 9, 14], skipinitialspace=True)
    train = pd.read_csv(directory / "adult.data", **options)
    test = pd.read_csv(directory / "adult.test", skiprows=1, **options)
    # MAF AdultDataset concatenates TEST, then TRAIN, and uses positional IDs.
    raw = pd.concat([test, train], ignore_index=True)
    raw.index = raw.index.astype(str)
    rows = raw.loc[dataset.instance_names]
    sex = (rows[9].str.strip() == "Male").to_numpy(dtype=float)
    race = (rows[8].str.strip() == "White").to_numpy(dtype=float)
    income = (rows[14].str.strip().str.rstrip(".") == ">50K").to_numpy(dtype=float)
    for name, values in (("sex", sex), ("race", race)):
        index = dataset.protected_attribute_names.index(name)
        if not np.array_equal(values, dataset.protected_attributes[:, index]):
            raise ValueError("Adult sensitive attributes do not align with MAF instance IDs.")
    if not np.array_equal(income, dataset.labels.ravel()):
        raise ValueError("Adult raw labels do not align with MAF instance IDs.")
    age = (pd.to_numeric(rows[0]) >= 40).to_numpy(dtype=float)
    married = rows[5].str.strip().isin(["Married-civ-spouse", "Married-AF-spouse"]).to_numpy(dtype=float)
    result = dataset.copy(deepcopy=True)
    additions = np.column_stack([age, married])
    result.protected_attributes = np.column_stack([result.protected_attributes, additions])
    result.protected_attribute_names += ["age", "marital-status"]
    result.privileged_protected_attributes += [np.array([1.]), np.array([1.])]
    result.unprivileged_protected_attributes += [np.array([0.]), np.array([0.])]
    # Upstream concat mode includes sensitive attributes in predictor inputs.
    result.features = np.column_stack([result.features, additions])
    result.feature_names += ["age", "marital-status"]
    result.metadata["protected_attribute_maps"] += [
        {0.: "Under 40", 1.: "40 or older"}, {0.: "Not married", 1.: "Married"}]
    result.validate_dataset()
    return result
