#!/usr/bin/env python
# coding: utf-8

# # 📘 Reusable Machine Learning - Regression Models
# This notebook contains reusable code blocks for fitting and evaluating regression models.

# In[ ]:


from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


# ### Linear Regression

# When is it used?    Linear relationships between variables

# In[ ]:


def linear_regression(df, target_col='target'):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("🔹 Linear Regression")
    print("R² Score:", r2_score(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    return model


# ### Ridge Regression

# When to use? -When multicolinearity is present

# In[ ]:


def ridge_regression(df, target_col='target', alpha=1.0):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"🔹 Ridge Regression (alpha={alpha})")
    print("R² Score:", r2_score(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    return model


# ### Lasso Regrssion

# When to use? -when you only want to pick the besh features that are most important for your model.

# In[ ]:


def lasso_regression(df, target_col='target', alpha=1.0):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Lasso(alpha=alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"🔹 Lasso Regression (alpha={alpha})")
    print("R² Score:", r2_score(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    return model


# ### Polynomial Regression

# In[ ]:


def polynomial_regression(df, target_col='target', degree=2):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"🔹 Polynomial Regression (degree={degree})")
    print("R² Score:", r2_score(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    return model,X_train,X_test,y_train,y_test


# ### Elastic Net Reggresion

# In[ ]:


def elasticnet_regression(df, target_col='target', alpha=1.0, l1_ratio=0.5):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)


    print("ElasticNet Regression Results:")
    print(f"MSE:{mean_squared_error(y_test, y_pred)}")
    print(f"R2 Score:{r2_score(y_test, y_pred)}")
    return model


# ### Random Forest Regressor

# When to Use?

#  Use when your data has nonlinear patterns, noise, or missing values.  
#     Math: Averages multiple decision trees (bagging) to reduce variance.  
#     Formula: Prediction = mean(Tree₁(x), Tree₂(x), ..., Treeₙ(x))  
#     Strength: Less overfitting due to aggregation (low variance).  

# In[ ]:


def run_random_forest_regressor(df, target_col='target', n_estimators=100, max_depth=None):
    """

    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Random Forest Regressor Results:")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")

    return model


# ### Gradient Boosting

# When to use?  
#     Use when: you want top accuracy and can tune parameters for complex data.  
#     Math: Builds trees sequentially to fix previous errors (boosting).  
#     Formula: Final Prediction = Σ (treeᵢ(x) * learning_rate)  
#     Strength: Low bias and low error due to focused correction.  

# In[ ]:


def run_gradient_boosting_regressor(df, target_col='target', n_estimators=100, learning_rate=0.1, max_depth=3):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Gradient Boosting Regressor Results:")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")

    return model


# ### How to call this folder?

# 

# In[ ]:




