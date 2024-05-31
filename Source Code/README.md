To set up the environment, follow these steps:

1. **Create and activate the conda environment**
```sh
conda create -n IT3190E_Group_4 python=3.10.12
conda activate IT3190E_Group_4
```

2. **Install the necessary packages**
```sh
pip install -r requirements.txt
```

- data_processing.ipynb dùng để phân loại giá vào các nhãn

- thư mục hyperparameter tuning dùng để tìm ra siêu tham số tốt nhất của các model

- model.ipynb tổng hợp các siêu tham số và đánh giá các mô hình

- demo.ipynb dùng để demo