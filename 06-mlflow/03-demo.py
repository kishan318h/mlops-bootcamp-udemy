import os
import mlflow
import argparse
import time

def eval(p1, p2):
    'dummy evaluation function created for demo'
    output_metric = p1**2 + p2**2
    return output_metric


def main(inp1, inp2):
    print("Tracking URI:", mlflow.get_tracking_uri())
    mlflow.set_experiment('Demo_Experiment') # create experiment

    with mlflow.start_run(run_name='test_run_1'):
        mlflow.set_tag("version","1.0.0")

        print('Logging params..')
        mlflow.log_param('param1', inp1) # log parameter
        mlflow.log_param('param2', inp2)

        metric = eval(p1 = inp1, p2 = inp2)
        print('logging metrics..')
        mlflow.log_metric('eval_metric', metric) # log metric

        # log artifact - create a dummy folder with a file named example.txt
        os.makedirs('dummy', exist_ok=True)
        with open('dummy/example.txt', 'wt') as f:
            f.write(f'Artifact created at {time.asctime()}')
        mlflow.log_artifact('dummy')


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--param1', '-p1', type=int, default=5)
    args.add_argument('--param2', '-p2', type=int, default=10)
    parsed_args = args.parse_args()

    # passing the arguments to the function
    main(parsed_args.param1, parsed_args.param2)

    ## to pass values of arguments
    ## python file.py --param1 val1 --param2 val2

