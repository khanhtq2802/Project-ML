import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer

class PricePredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Price Prediction App")
        
        # Create labels and entry widgets for features
        self.features = ['room', 'area', 'x', 'y', 'khoang_cach', 'n_hospital']
        self.entries = {}
        
        for feature in self.features:
            label = tk.Label(root, text=feature)
            label.pack()
            entry = tk.Entry(root)
            entry.pack()
            self.entries[feature] = entry
        
        # Button to load model file
        self.load_button = tk.Button(root, text="Load Model File", command=self.load_model)
        self.load_button.pack()
        
        # Button to make prediction
        self.predict_button = tk.Button(root, text="Predict Price", command=self.predict_price)
        self.predict_button.pack()
        
        # Label to display the prediction
        self.result_label = tk.Label(root, text="")
        self.result_label.pack()
        
        self.model = None
        self.feature_scaler = None
        self.target_scaler = None
        self.selected_columns = None
        
    def load_model(self):
        # Open file dialog to select model file
        model_file = filedialog.askopenfilename(title="Select Model File", filetypes=(("PKL files", "*.pkl"),))
        if model_file:
            try:
                # Load the model and scalers
                self.model = joblib.load(model_file)
                self.feature_scaler = joblib.load('/home/khanh/projects/Project-ML/GUI/feature_scaler.pkl')
                self.target_scaler = joblib.load('/home/khanh/projects/Project-ML/GUI/target_scaler.pkl')
                # Load the selected columns used during training
                self.selected_columns = joblib.load('/home/khanh/projects/Project-ML/GUI/selected_columns.pkl')
                messagebox.showinfo("Success", "GUIModel and scalers loaded successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model or scalers. Error: {e}")
    
    def predict_price(self):
        if self.model is None or self.feature_scaler is None or self.target_scaler is None or self.selected_columns is None:
            messagebox.showwarning("Warning", "Please load a model, scalers, and selected columns first.")
            return
        
        try:
            # Collect input data
            input_data = {feature: float(self.entries[feature].get()) for feature in self.features}
            input_df = pd.DataFrame([input_data])
            
            # One-hot encode categorical variables in the input data
            input_en = pd.get_dummies(input_df)
            
            # Align input data with selected columns
            input_en = input_en.reindex(columns=self.selected_columns, fill_value=0)
            
            # Scale the input data
            input_scaled = self.feature_scaler.transform(input_en)
            
            # Predict the price
            prediction_scaled = self.model.predict(input_scaled)
            prediction_scaled = prediction_scaled.reshape(-1, 1)
            
            # Inverse transform the prediction to get the original scale
            prediction = self.target_scaler.inverse_transform(prediction_scaled)
            
            # Display the prediction
            self.result_label.config(text=f"Predicted Price: {prediction[0][0]:.2f}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to make prediction. Error: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PricePredictionApp(root)
    root.mainloop()