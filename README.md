# Aranyani2.0

This project is based on code from:
https://github.com/brcsomnath/Aranyani
Licensed under the MIT License.

python -m src.main --dataset adult --batch_size 1 --lambda_const 1.0 --use_test_set=True --use_class_weights=False --load_model=True --model_path='models/adult_model' --prequential=True --drift=True --drift_scenario='female_income_parity'