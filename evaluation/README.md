# Evaluating In-context Few-shot Learners

To evaluate the in-context few-shot learners, we provide a `multilingual_evaluator`
script with multiple configurable parameters.

The evaluation of your desired model can be run as follows:

```shell
pip install -r evaluation/requirements.txt
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

CUDA_VISIBLE_DEVICES={device_id} python evaluation/multilingual_evaluator.py --model_names_or_paths {HF_model_id} --langs cs,ru,pl --firstn 10
```
See the corresponding script, or `python evaluation/multilingual_evaluator.py -h` for a full list of parameters.

Using our uploaded models, you can also reproduce the results reported in the referenced paper.

Note that this evaluation is a part of a larger project, 
which is why the evaluation scripts may contain some functionality 
that is not directly relevant here. 
If you'd like to perform some changes in the evaluation routine, 
it might be easier to start from scratch - see the project root README for a simple example. 
