from numpy import mean, std
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score


def nested_cv(X, y, model, grid_search_space, inner_splits=3, outer_splits=10, random_state=1):
    print(X.shape)
    print(y.shape)
    print(model)
    print(grid_search_space)
    cv_inner = KFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
    search = GridSearchCV(model, grid_search_space, scoring='accuracy', n_jobs=4, cv=cv_inner, refit=True, verbose=10)
    cv_outer = KFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(search, X, y, scoring='accuracy', cv=cv_outer, n_jobs=-1)
    return (mean(scores), std(scores))
