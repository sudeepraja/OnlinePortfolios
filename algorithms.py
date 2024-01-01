import pandas as pd
import cvxpy as cvx
import numpy as np

from scipy.special import logsumexp



# Uniform Constant Rebalanced Portfolio
def UCRP(df):
    rows, cols = df.shape
    weights = pd.DataFrame(index=df.index, columns=df.columns)
    # repeat the same weights for all days
    weights.iloc[:] = 1 / cols
    return weights

# Best Constant Rebalanced Portfolio
def BCRP(df):
    rows, cols = df.shape
    x = cvx.Variable(cols, nonneg=True) 

    # objective function
    # Matrix vector multiplication
    mat = df.values
    obj = cvx.log(mat @ x)
    obj = cvx.sum(obj)

    # constraints
    constraints = [cvx.sum(x) == 1]

    # solve the problem
    prob = cvx.Problem(cvx.Maximize(obj), constraints)
    prob.solve(solver='SCS')

    # return the weights
    weights = pd.DataFrame(index=df.index, columns=df.columns)
    weights.iloc[:] = x.value
    return weights

# Follow the Leader
def FTL(df):
    rows, cols = df.shape
    x = cvx.Variable(cols, nonneg=True)

    weights = pd.DataFrame(index=df.index, columns=df.columns)
    weights.iloc[0] = 1 / cols

    for i in range(1,rows):
        # objective function
        # Matrix vector multiplication
        mat = df.iloc[:i].values
        obj = cvx.log(mat @ x)
        obj = cvx.sum(obj)

        # constraints
        constraints = [cvx.sum(x) == 1]

        # solve the problem
        prob = cvx.Problem(cvx.Maximize(obj), constraints)
        prob.solve(solver='SCS')

        weights.iloc[i] = x.value

    return weights



# Smooth Prediction
# Hazana Agarwal
# FTL with log barrier
def Smooth_Prediction(df):
    rows, cols = df.shape
    x = cvx.Variable(cols, nonneg=True)

    weights = pd.DataFrame(index=df.index, columns=df.columns)
    weights.iloc[0] = 1 / cols

    for i in range(1,rows):
        # objective function
        # Matrix vector multiplication
        mat = df.iloc[:i].values
        obj = cvx.log(mat @ x)
        obj = cvx.sum(obj)
        obj = obj + cvx.sum(cvx.log(x))


        # constraints
        constraints = [cvx.sum(x) == 1]

        # solve the problem
        prob = cvx.Problem(cvx.Maximize(obj), constraints)
        prob.solve(solver='SCS')

        weights.iloc[i] = x.value

    return weights

# Exp-Concave FTL
# Hazan Kale - AN ONLINE PORTFOLIO SELECTION ALGORITHM WITH REGRET LOGARITHMIC IN PRICE VARIATION
# FTL with 2norm regularizer
def Exp_Concave_FTL(df):
    rows, cols = df.shape
    x = cvx.Variable(cols, nonneg=True)

    weights = pd.DataFrame(index=df.index, columns=df.columns)
    weights.iloc[0] = 1 / cols

    for i in range(1,rows):
        # objective function
        # Matrix vector multiplication
        mat = df.iloc[:i].values
        obj = cvx.log(mat @ x)
        obj = cvx.sum(obj)
        # 2norm squared regularizer
        obj = obj - cvx.sum_squares(x)/2


        # constraints
        constraints = [cvx.sum(x) == 1]

        # solve the problem
        prob = cvx.Problem(cvx.Maximize(obj), constraints)
        prob.solve(solver='SCS')

        weights.iloc[i] = x.value

    return weights

# Be the Leader
def BTL(df):
    rows, cols = df.shape
    x = cvx.Variable(cols, nonneg=True)

    weights = pd.DataFrame(index=df.index, columns=df.columns)

    for i in range(0,rows):
        # objective function
        # Matrix vector multiplication
        mat = df.iloc[:i+1].values
        obj = cvx.log(mat @ x)
        obj = cvx.sum(obj)

        # constraints
        constraints = [cvx.sum(x) == 1]

        # solve the problem
        prob = cvx.Problem(cvx.Maximize(obj), constraints)
        prob.solve(solver='SCS')

        weights.iloc[i] = x.value

    return weights

def exp_normalize(x):
    b = x.max()
    y = np.exp(x - b)
    return y / y.sum()

def AdaEG(df):
    rows, cols = df.shape
    # divide each row by the maximum value

    weights = pd.DataFrame(index=df.index, columns=df.columns)

    lr_term = 1
    gradient_sum = 0

    x_t_plus_1 = np.ones(cols)/cols

    for t in range(rows):
        weights.iloc[t] = x_t_plus_1

        gradient = - df.iloc[t]/np.dot(df.iloc[t],x_t_plus_1)
        gradient_sum = gradient_sum + gradient

        lr_term = lr_term + np.dot(np.multiply(gradient,(1-x_t_plus_1)/(-np.log(x_t_plus_1))),gradient)
        eta = np.sqrt(np.log(cols)/lr_term)

        x_t_plus_1 = exp_normalize(eta*gradient_sum)

    return weights


def EG(df, eta=0.05):
    rows, cols = df.shape

    weights = pd.DataFrame(index=df.index, columns=df.columns)

    x_t_plus_1 = np.ones(cols)/cols
    theta = 0

    for t in range(rows):
        weights.iloc[t] = x_t_plus_1

        gradient = - df.iloc[t]/np.dot(df.iloc[t],x_t_plus_1)
        theta = theta - gradient

        x_t_plus_1 = exp_normalize(eta*theta)

    return weights

def SoftBayes(df):
    rows, cols = df.shape

    weights = pd.DataFrame(index=df.index, columns=df.columns)

    x_t_plus_1 = np.ones(cols)/cols

    for t in range(rows):
        weights.iloc[t] = x_t_plus_1

        eta_t  = np.sqrt(np.log(cols)/ (2*cols * (t+1)))
        eta_t_plus_1 = np.sqrt(np.log(cols)/ (2*cols * (t+2)))

        x_t_plus_1 = x_t_plus_1 * (1-eta_t + eta_t*df.iloc[t]/ np.dot(df.iloc[t],x_t_plus_1))*eta_t_plus_1/eta_t + (1-eta_t_plus_1/eta_t)*1/cols
    return weights

def Implicit_Equation_solver(l):
    ## Solves for x and lambda such that
    ## x_i = 1/(l_i + lambda), i = 1,2,...,n
    ## sum(x_i) = 1

    ## Perform binary searh on lambda
    ## to find the smallest lambda such that
    ## x_i >= 0 for all i and sum(x_i) = 1

    LB = - min(l) # if lambda = LB the some x_i = inf
    
    func = lambda x: sum(1/(l+x))

    # Find some y such that func(y) < 1
    y = max(0, LB)+1
    while func(y) > 1:
        y = 2*y

    UB = y

    while UB - LB > 1e-8:
        mid = (UB + LB)/2
        if func(mid) > 1:
            LB = mid
        else:
            UB = mid

    lambda_star = (UB + LB)/2
    x_star = 1/(l + lambda_star)
    return x_star/sum(x_star)


def LogBarrier(df):
    rows, cols = df.shape

    weights = pd.DataFrame(index=df.index, columns=df.columns)

    x_t_plus_1 = np.ones(cols)/cols
    lr_term = 1
    gradient_sum = 0

    for i in range(0,rows):
        weights.iloc[i] = x_t_plus_1

        gradient = - df.iloc[i].values/np.dot(df.iloc[i].values,x_t_plus_1)
        gradient_sum = gradient_sum + gradient
        # Hadaamard product of gradient and x_t_plus_1
        lr_term = lr_term + np.dot(np.multiply(gradient,x_t_plus_1),np.multiply(gradient,x_t_plus_1))
        eta = np.sqrt(cols)/np.sqrt(lr_term + 4*cols)
        
        # print(i,lr_term)

        x_t_plus_1 = Implicit_Equation_solver(gradient_sum*eta)

    return weights


#Quadratize and no regularization
def FTAL(df):
    rows, cols = df.shape

    # Transform the df such that the maximum value of each row is 1
    df_ = df.div(df.max(axis=1), axis=0)

    weights = pd.DataFrame(index=df.index, columns=df.columns)
    

    x_t_plus_1 = np.ones(cols)/cols
    linear_term = 0
    quadratic_term = 0

    for t in range(rows):
        weights.iloc[t] = x_t_plus_1

        linear_term = linear_term - df_.iloc[t].values - df_.iloc[t].values/np.dot(df_.iloc[t].values,x_t_plus_1)
        quadratic_term = quadratic_term + np.outer(df_.iloc[t].values,df_.iloc[t].values)/np.dot(df_.iloc[t].values,x_t_plus_1)

        x = cvx.Variable(cols, nonneg=True)
        # objective function
        obj = linear_term @ x + cvx.quad_form(x,cvx.psd_wrap(quadratic_term))
        # constraints
        constraints = [cvx.sum(x) == 1]

        # solve the problem
        prob = cvx.Problem(cvx.Minimize(obj), constraints)
        # Mention the solver explicitly and say that it is a quadratic problem with positive definite matrix
        prob.solve(solver='OSQP', qcp=True)

        x_t_plus_1 = x.value

    return weights

# Quadratize and 2norm regularization
def FTRL(df):
    rows, cols = df.shape

    # Transform the df such that the maximum value of each row is 1
    df_ = df.div(df.max(axis=1), axis=0)

    weights = pd.DataFrame(index=df.index, columns=df.columns)
    

    x_t_plus_1 = np.ones(cols)/cols
    linear_term = 0
    quadratic_term = np.eye(cols)/2

    for t in range(rows):
        weights.iloc[t] = x_t_plus_1

        linear_term = linear_term - df_.iloc[t].values - df_.iloc[t].values/np.dot(df_.iloc[t].values,x_t_plus_1)
        quadratic_term = quadratic_term + np.outer(df_.iloc[t].values,df_.iloc[t].values)/np.dot(df_.iloc[t].values,x_t_plus_1)

        x = cvx.Variable(cols, nonneg=True)
        # objective function
        obj = linear_term @ x + cvx.quad_form(x,cvx.psd_wrap(quadratic_term))
        # constraints
        constraints = [cvx.sum(x) == 1]

        # solve the problem
        prob = cvx.Problem(cvx.Minimize(obj), constraints)
        prob.solve(solver='OSQP', qcp=True)

        x_t_plus_1 = x.value

    return weights
