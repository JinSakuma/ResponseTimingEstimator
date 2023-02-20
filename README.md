# ResponseTimingEstimator

Response Timing Estimator for "Response Timing Estimation for Spoken Dialog Systems based on Syntactic Completeness Prediction"

## Instllation
* Create conda env: ```conda create -n torch17 -f torcha7.yml```
* export python path: ``` . path.sh ```

## Training
VAD Model
```
python script/vad/run_vad.py config=<config path> --gpu_id <gpu id>
```

Language Model
``` 
python script/lm/run_lm_lstm.py config=<config path> --gpu_id <gpu id>
```

Resoinse Timing Estimator
``` 
python script/timing/run_timing.py config=<config path> --model <baseline / proposed> --cv_id <cross validation id> --gpu_id <gpu id>
```

## Reference
```
@article{Sakuma2023ResponseTE,
  title={Response Timing Estimation for Spoken Dialog Systems Based on Syntactic Completeness Prediction},
  author={Jin Sakuma and Shinya Fujie and Tetsunori Kobayashi},
  journal={2022 IEEE Spoken Language Technology Workshop (SLT)},
  year={2023},
  pages={369-374}
}
```
