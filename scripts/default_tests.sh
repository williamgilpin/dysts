#!/bin/bash

python rc_chaos/Methods/Models/esn/esn_rc_dyst_copy.py --reservoir_size 1000 --sparsity 0.01 --radius 0.6 --sigma_input 1 --dynamics_fit_ratio 0.2857 --regularization 0.0 --scaler_tt Standard --solver auto

# python rc_chaos/Methods/Models/esn/esn_rc_dyst.py esn_rc_dyst --mode rc_dyst --display_output 0 --system_name Lorenz3D --write_to_log 1 --N 100000 --N_used 1000 --RDIM 1 --noise_level 10 --scaler Standard --approx_reservoir_size 1000 --degree 10 --radius 0.6 --sigma_input 1 --regularization 0.0 --dynamics_length 200 --iterative_prediction_length 500 --num_test_ICS 2 --solver auto --number_of_epochs 1000000 --learning_rate 0.001 --reference_train_time 10 --buffer_train_time 0.5