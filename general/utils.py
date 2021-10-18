import shap
from captum.attr import DeepLiftShap
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
import xgboost
from collections import Counter


def aux_counter(lista, names, dim):
    total_list = []

    for i_i in range(len(lista)):
        prov = []
        for j in range(dim):
            d = Counter(lista[i_i])
            if j in lista[i_i]:
                prov.append(d[j])
            else:
                prov.append(0)
        total_list.append(prov)
    total = pd.DataFrame(total_list, columns=names)
    total_sum = total.sum(axis=0)
    return pd.DataFrame(total_sum)


def xgboost_preprocessing(train_dataset, colors_trainset, test_dataset, colors_testset):
    # Tensor2DF
    train_df = pd.DataFrame(train_dataset.detach().numpy())
    test_df = pd.DataFrame(test_dataset.detach().numpy())

    # Create all dataset
    entire_dataset = pd.concat((train_df, test_df), ignore_index=True)
    entire_colors = np.concatenate((colors_trainset, colors_testset))

    # Split on groups and clean outliers
    entire_shh = entire_dataset.loc[entire_colors == 'SHH']
    entire_wnt = entire_dataset.loc[entire_colors == 'WNT']
    entire_g3 = entire_dataset.loc[entire_colors == 'Group3']
    entire_g4 = entire_dataset.loc[entire_colors == 'Group4']

    outliers_shh = find_outliers(entire_shh, 0.1)
    outliers_wnt = find_outliers(entire_wnt, 0.1)
    outliers_g3 = find_outliers(entire_g3, 0.1)
    outliers_g4 = find_outliers(entire_g4, 0.1)
    outliers_total = outliers_wnt.union(outliers_shh.union(outliers_g3.union(outliers_g4)))

    entire_data_outliers = entire_dataset.drop(list(outliers_total))
    entire_colors_outliers = pd.DataFrame(entire_colors).drop(list(outliers_total))

    entire_data_outliers = torch.tensor(entire_data_outliers.values).float()
    entire_colors_outliers = entire_colors_outliers.to_numpy().reshape(len(entire_colors_outliers))

    # x_train, x_test, y_train, y_test = train_test_split(entire_data_outliers, entire_colors_outliers,
    #                                                      test_size=0.3,
    #                                                      random_state=1)

    return entire_data_outliers, entire_colors_outliers
    # return x_train, x_test, y_train, y_test


def get_embeddings(model, dataloader, device):
    model.eval()
    rec_model = np.zeros(shape=(0, model.decoder[2].out_features))
    embedding_model = np.zeros(shape=(0, model.features))
    mean_model = np.zeros(shape=(0, model.features))
    logvar_model = np.zeros(shape=(0, model.features))
    with torch.no_grad():  # in validation we don't want to update weights
        for data in dataloader:
            data = data.to(device)
            reconstruction, mean, logvar, coded = model(data)
            rec_model = np.concatenate((rec_model, reconstruction.to('cpu')), axis=0)
            mean_model = np.concatenate((mean_model, mean.to('cpu')), axis=0)
            logvar_model = np.concatenate((logvar_model, logvar.to('cpu')), axis=0)
            embedding_model = np.concatenate((embedding_model, coded.to('cpu')), axis=0)

    return rec_model, embedding_model, mean_model, logvar_model


def xgboost_shap(model, train_loader_beta, colors_coded_train,
                 test_loader_beta, colors_coded_test, latent_variables, device):
    # Get the embeddings
    _, coded_train_beta, _, _ = get_embeddings(model, train_loader_beta, device)
    _, coded_test_beta, _, _ = get_embeddings(model, test_loader_beta, device)

    print("Inside :", coded_train_beta.shape, len(colors_coded_train.ravel()))

    # Train the classifier
    xgboost_classifier_beta = xgboost.XGBClassifier(random_state=123)
    xgboost_classifier_beta.fit(coded_train_beta, colors_coded_train.ravel())

    y_pred = xgboost_classifier_beta.predict(coded_test_beta)

    accuracy = accuracy_score(colors_coded_test.ravel(), y_pred)
    print("Accuracy with all LV: ", accuracy)

    # SHAP Tree Explainer to get the importance
    whole_dataset = np.concatenate((coded_train_beta, coded_test_beta))
    explainer = shap.TreeExplainer(xgboost_classifier_beta)
    shap_values = explainer.shap_values(whole_dataset)  # we want to explain the whole dataset

    feature_importance_shap = np.sum(np.abs(shap_values).mean(1),
                                     axis=0)  # mean within class and summation between classes
    indices_beta_sort = np.argsort(feature_importance_shap)[::-1]

    # Top features
    indices_beta = indices_beta_sort[:latent_variables]

    coded_beta_filtered = pd.DataFrame(coded_train_beta)[indices_beta]
    coded_beta_filtered_test = pd.DataFrame(coded_test_beta)[indices_beta]

    xgboost_classifier_beta = xgboost.XGBClassifier()
    xgboost_classifier_beta.fit(coded_beta_filtered, colors_coded_train.ravel())

    y_pred = xgboost_classifier_beta.predict(coded_beta_filtered_test)

    accuracy = accuracy_score(colors_coded_test, y_pred)
    print("Accuracy with X LV: ", accuracy)

    return indices_beta


def shap_vae(model, examples, shap_indices, device):

    background = examples[:150]
    test_shap = examples[150:]

    net = model.encoder.to(device)
    dl = DeepLiftShap(net, True)
    genes = []
    for ind in shap_indices:
        attribution = dl.attribute(test_shap.to(device), baselines=background.to(device), target=int(ind))
        attribution = attribution.detach()
        attribution_mean_tensor = torch.mean(attribution, dim=0)
        for element in range(len(attribution_mean_tensor)):
            if attribution_mean_tensor[element] > torch.mean(attribution_mean_tensor) + torch.std(attribution_mean_tensor) * 3:
                genes.append(element)
            if attribution_mean_tensor[element] < torch.mean(attribution_mean_tensor) - torch.std(attribution_mean_tensor) * 3:
                genes.append(element)

    return genes


def find_outliers(data, percentage=0.10):
    candidates_final = []
    return_list = []

    for column in data.columns:
        q1 = data[column].quantile(0.25)
        q3 = data[column].quantile(0.75)
        iqr = q3 - q1
        candidates_final = candidates_final + (list(data.loc[data[column] < (q1 - 1.5 * iqr)].index))
        candidates_final = candidates_final + (list(data.loc[data[column] > (q3 + 1.5 * iqr)].index))

    for i in candidates_final:
        if candidates_final.count(i) > percentage * len(data.columns):
            return_list.append(i)

    return set(return_list)