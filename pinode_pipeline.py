import torch
import torch.nn as nn
import pandas as pd
from torchdiffeq import odeint
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def load_data(path):
    df = pd.read_excel(path, sheet_name='ModPlant', header=1)
    return df

def preprocess(df):
    num_cols = [
        'Lifetime Avg [N]', 'Lifetime Avg [P]', 'Lifetime Avg [K]',
        'Growth Day', 'Treatment %',
        'Lifetime Average [Ca]', 'Lifetime Average [Mg]',
        'Lifetime Average [S]', 'Lifetime Average [Fe]',
        'Harvest [N] (mg/L)', 'Harvest [P] (mg/L)', 'Harvest [K] (mg/L)'
    ]

    cat_cols = ['Limiting Nutrient']

    df = df.replace(r'^\s*$', None, regex=True)

    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=num_cols + cat_cols + ['Lifetime Mu', 'Harvest Mu'])

    y = df[['Lifetime Mu', 'Harvest Mu']].values

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(sparse_output=False), cat_cols)
    ])

    X = preprocessor.fit_transform(df[num_cols + cat_cols])

    return train_test_split(X, y, test_size=0.2, random_state=42), preprocessor

class ODEFunc(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        self.mu_max = nn.Parameter(torch.tensor(0.5))
        self.Ks = nn.Parameter(torch.tensor(1.0))
        self.m = nn.Parameter(torch.tensor(0.01))
        self.d = nn.Parameter(torch.tensor(10.0))
        self.KI = nn.Parameter(torch.tensor(100.0))
        self.theta = nn.Parameter(torch.tensor(0.05))

    def forward(self, t, state):
        X, W = state
        S = X[:, :3].mean(dim=1, keepdim=True)
        I = torch.ones_like(S) * 200
        T = torch.ones_like(S) * 298

        f_I = I / (I + self.KI)
        g_T = torch.exp(self.theta * (T - 298))

        monod = self.mu_max * f_I * g_T * (S / (S + self.Ks))
        maturity = self.m * (t - self.d)

        eps = self.net(torch.cat([X, t.expand(X.size(0), 1)], dim=1))

        mu = monod + maturity + eps
        dWdt = mu * W

        return (torch.zeros_like(X), dWdt)

def physics_loss(model, X):
    X = torch.tensor(X, dtype=torch.float32)
    W = torch.ones(X.size(0), 1)
    t = torch.tensor(10.0)

    S = X[:, :3].mean(dim=1, keepdim=True)
    monod = model.mu_max * (S / (S + model.Ks))
    maturity = model.m * (t - model.d)

    eps = model.net(torch.cat([X, t.expand(X.size(0), 1)], dim=1))
    mu = monod + maturity + eps

    dWdt_pred = mu * W
    dWdt_true = mu.detach() * W

    return nn.MSELoss()(dWdt_pred, dWdt_true)

def train_model(X_train, y_train, input_dim):
    model = ODEFunc(input_dim)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    t = torch.linspace(0, 30, steps=10)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(200):
        optimizer.zero_grad()
        W0 = torch.ones(X_train.size(0), 1)
        pred = odeint(model, (X_train, W0), t, method='dopri5')[1][-1]

        data_loss = nn.MSELoss()(pred.squeeze(), y_train[:, 0])
        phys_loss = physics_loss(model, X_train)

        loss = data_loss + 0.1 * phys_loss

        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.1)

    def closure():
        optimizer.zero_grad()
        W0 = torch.ones(X_train.size(0), 1)
        pred = odeint(model, (X_train, W0), t, method='dopri5')[1][-1]
        loss = nn.MSELoss()(pred.squeeze(), y_train[:, 0])
        loss.backward()
        return loss

    optimizer.step(closure)

    return model

def evaluate(model, X_test, y_test):
    model.eval()

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    t = torch.linspace(0, 30, steps=10)
    W0 = torch.ones(X_test.size(0), 1)

    with torch.no_grad():
        pred = odeint(model, (X_test, W0), t, method='dopri5')[1][-1]

    pred = pred.squeeze().numpy()
    y_true = y_test[:, 0].numpy()

    mse = mean_squared_error(y_true, pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_true, pred)

    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")

    return pred, y_true

def plot_results(pred, y_true):
    plt.figure()
    plt.scatter(y_true, pred)
    plt.xlabel("Actual μ")
    plt.ylabel("Predicted μ")
    plt.title("Predicted vs Actual μ")
    plt.savefig("prediction_vs_actual.png")

    plt.figure()
    plt.plot(y_true, label="Actual")
    plt.plot(pred, label="Predicted")
    plt.legend()
    plt.title("Prediction Curve")
    plt.savefig("prediction_curve.png")

if __name__ == "__main__":
    path = "NPK.CrossT.All.xlsx"

    df = load_data(path)

    (X_train, X_test, y_train, y_test), preprocessor = preprocess(df)

    model = train_model(X_train, y_train, input_dim=X_train.shape[1])

    torch.save(model.state_dict(), "pinode_model.pth")

    pred, y_true = evaluate(model, X_test, y_test)

    plot_results(pred, y_true)

    print("Training, evaluation, and plotting complete.")