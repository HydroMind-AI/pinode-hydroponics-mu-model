import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

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

    return train_test_split(X, y, test_size=0.2, random_state=42)

class MuModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.mu_max = nn.Parameter(torch.tensor(0.3))
        self.Ks = nn.Parameter(torch.tensor(1.0))
        self.m = nn.Parameter(torch.tensor(0.01))
        self.d = nn.Parameter(torch.tensor(10.0))

    def forward(self, X, t):
        S = X[:, :3].mean(dim=1, keepdim=True)

        monod = self.mu_max * (S / (S + self.Ks))
        maturity = self.m * (t - self.d)

        eps = self.net(torch.cat([X, t], dim=1))

        mu = monod + maturity + eps

        return mu

def physics_loss(model, X, t):
    S = X[:, :3].mean(dim=1, keepdim=True)
    monod = model.mu_max * (S / (S + model.Ks))
    maturity = model.m * (t - model.d)

    mu_pred = model(X, t)
    mu_phys = monod + maturity

    return nn.MSELoss()(mu_pred, mu_phys.detach())

def train_model(X_train, y_train, input_dim):
    model = MuModel(input_dim)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    t = X_train[:, 3].unsqueeze(1)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(400):
        optimizer.zero_grad()

        mu_pred = model(X_train, t)

        data_loss = nn.MSELoss()(mu_pred.squeeze(), y_train[:, 0])
        phys_loss = physics_loss(model, X_train, t)

        loss = data_loss + 0.05 * phys_loss

        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.1)

    def closure():
        optimizer.zero_grad()
        mu_pred = model(X_train, t)
        loss = nn.MSELoss()(mu_pred.squeeze(), y_train[:, 0])
        loss.backward()
        return loss

    optimizer.step(closure)

    return model

def evaluate(model, X_test, y_test):
    model.eval()

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    t = X_test[:, 3].unsqueeze(1)

    with torch.no_grad():
        pred = model(X_test, t).squeeze().numpy()

    pred = np.clip(pred, 0, None)
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
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()],
             linestyle='--')
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

    X_train, X_test, y_train, y_test = preprocess(df)

    model = train_model(X_train, y_train, input_dim=X_train.shape[1])

    torch.save(model.state_dict(), "pinode_model.pth")

    pred, y_true = evaluate(model, X_test, y_test)

    plot_results(pred, y_true)

    print("Done.")