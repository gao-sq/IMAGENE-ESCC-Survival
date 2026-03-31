import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def train_model(train_data, train_labels, val_data, val_labels, 
                use_class_weight=True, use_smote=True, random_state=0,
                model_save_path='model.pkl'
                ):
    
    # Calculate class weights
    class_weights_dict = None
    if use_class_weight:
        class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
        class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
    
    # Use SMOTE for oversampling
    if use_smote:
        smote = SMOTE(random_state=0)
        train_data, train_labels = smote.fit_resample(train_data, train_labels)
    
    clf = HistGradientBoostingClassifier(
                                        max_iter=100,
                                        max_depth=5,
                                        min_samples_leaf=40,
                                        l2_regularization=1.0,
                                        # early_stopping=True,  # Use early stopping
                                        validation_fraction=0.25,  # Validation set ratio
                                        n_iter_no_change=10,  # Number of iterations for early stopping
                                        random_state=random_state,
                                        class_weight=class_weights_dict
                                        )
    clf.fit(train_data, train_labels)
    train_pred = clf.predict(train_data)
    val_pred = clf.predict(val_data)

    train_acc = accuracy_score(train_labels, train_pred)
    val_acc = accuracy_score(val_labels, val_pred)

    if model_save_path:
        joblib.dump(clf, model_save_path)

    return clf, train_pred, val_pred, train_acc, val_acc