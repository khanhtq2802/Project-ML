import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import numpy as np

class PredictApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Predict App")

        self.model_options = [
            "Gradient Boosting",
            "KNN",
            "Random Forest",
            "SVM",
            "XGBoost"
        ]

        self.model_files = {
            "Gradient Boosting": "/media/khanh/Data/projects/Project-ML/classification/weights/gradient_boosting_model.pkl",
            "KNN": "/media/khanh/Data/projects/Project-ML/classification/weights/knn_model.pkl",
            "Random Forest": "/media/khanh/Data/projects/Project-ML/classification/weights/randomforest.pkl",
            "SVM": "/media/khanh/Data/projects/Project-ML/classification/weights/svm_model.pkl",
            "XGBoost": "/media/khanh/Data/projects/Project-ML/classification/weights/xgboost_model.pkl"
        }

        self.scaler_file = "/media/khanh/Data/projects/Project-ML/classification/weights/scaler.pkl"
        self.create_widgets()

    def create_widgets(self):
        # Dropdown menu for model selection
        self.model_label = tk.Label(self.root, text="Select Model:")
        self.model_label.grid(row=0, column=0, padx=10, pady=10)
        self.model_var = tk.StringVar()
        self.model_menu = ttk.Combobox(self.root, textvariable=self.model_var, values=self.model_options)
        self.model_menu.grid(row=0, column=1, padx=10, pady=10)

        # Entry fields for input data
        self.entries = {}
        self.labels = [
            "area", "x", "y", "khoang_cach"
        ]

        for idx, label in enumerate(self.labels):
            tk.Label(self.root, text=label).grid(row=idx+1, column=0, padx=10, pady=5, sticky='w')
            self.entries[label] = tk.Entry(self.root)
            self.entries[label].grid(row=idx+1, column=1, padx=10, pady=5)

        # Combobox for boolean fields
        self.boolean_labels = {
            "hospital": ["n_hospital_0", "n_hospital_1", "n_hospital_2", "n_hospital_3", "n_hospital_4", "n_hospital_5"],
            "room": ["room_1", "room_2", "room_3", "room_4", "room_5"],
            "toilet": ["toilet_1", "toilet_2", "toilet_3", "toilet_4", "toilet_5"],
            "quan": [
                "quan_Ba Dinh", "quan_Bac Tu Liem", "quan_Cau Giay", "quan_Dan Phuong",
                "quan_Dong Anh", "quan_Dong Da", "quan_Gia Lam", "quan_Ha Dong",
                "quan_Hai Ba Trung", "quan_Hoai Duc", "quan_Hoang Mai", "quan_Long Bien",
                "quan_Nam Tu Liem", "quan_Tay Ho", "quan_Thanh Tri", "quan_Thanh Xuan", "quan_Vi Thanh"
            ],
            "polistic": ["polistic_chua so", "polistic_hdmb", "polistic_so do"],
            "furniture": ["furniture_cao cap", "furniture_co ban", "furniture_day du", "furniture_nguyen_ban"],
            "direct": [
                "direct_Bac", "direct_Dong", "direct_Dong - Bac", "direct_Dong - Nam",
                "direct_Nam", "direct_Tay", "direct_Tay - Bac", "direct_Tay - Nam"
            ],
            "direct2": [
                "direct2_Bac", "direct2_Dong", "direct2_Dong - Bac", "direct2_Dong - Nam",
                "direct2_Nam", "direct2_Tay", "direct2_Tay - Bac", "direct2_Tay - Nam"
            ]
        }

        for category, options in self.boolean_labels.items():
            tk.Label(self.root, text=category).grid(row=len(self.entries)+1, column=0, padx=10, pady=5, sticky='w')
            self.entries[category] = ttk.Combobox(self.root, values=options)
            self.entries[category].grid(row=len(self.entries)+1, column=1, padx=10, pady=5)

        # Predict button
        self.predict_button = tk.Button(self.root, text="Predict", command=self.predict)
        self.predict_button.grid(row=len(self.entries)+1, column=0, columnspan=2, pady=10)

        # Prediction result
        self.result_label = tk.Label(self.root, text="Prediction Result:")
        self.result_label.grid(row=len(self.entries)+2, column=0, padx=10, pady=10)
        self.result_var = tk.StringVar()
        self.result_entry = tk.Entry(self.root, textvariable=self.result_var, state='readonly')
        self.result_entry.grid(row=len(self.entries)+2, column=1, padx=10, pady=10)

    def load_model(self, model_name):
        return joblib.load(self.model_files[model_name])

    def load_scaler(self):
        return joblib.load(self.scaler_file)

    def predict(self):
        try:
            model_name = self.model_var.get()
            if not model_name:
                messagebox.showerror("Error", "Please select a model")
                return
            
            # Load the model
            model = self.load_model(model_name)
            
            # Collect input data
            input_data = []
            for label in self.labels:
                value = self.entries[label].get()
                if value == "":
                    messagebox.showerror("Error", f"Please enter value for {label}")
                    return
                input_data.append(float(value))
            
            # Collect boolean data
            for category, options in self.boolean_labels.items():
                value = self.entries[category].get()
                boolean_data = [1 if option == value else 0 for option in options]
                input_data.extend(boolean_data)
            
            input_data = np.array(input_data).reshape(1, -1)

            # Scale data if necessary
            if model_name in ["KNN", "SVM"]:
                scaler = self.load_scaler()
                input_data = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_data)

            string = ''
            if prediction[0] <= 20:
                string = 'price <= 20'
            elif prediction[0] <= 30:
                string = '20 < price <= 30'
            elif prediction[0] <= 45:
                string = '30 < price <= 45'
            elif prediction[0] <= 60:
                string = '45 < price <= 60'
            else:
                string = '60 < price'
            self.result_var.set(string)

        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = PredictApp(root)
    root.mainloop()
