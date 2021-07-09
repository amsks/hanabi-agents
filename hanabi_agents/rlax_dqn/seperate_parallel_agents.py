import pickle
import haiku as hk 
import names 
import os
import numpy as np

def main(pkl_path = None,
         agent_type = 'rainbow',
         save_to = None):

    
    with open(pkl_path, 'rb') as iwf:
        base = pickle.load(iwf)
        base_mutable = hk.data_structures.to_mutable_dict(base)
        x = 0
        for m in base_mutable:

            base_mutable[m] = hk.data_structures.to_mutable_dict(base_mutable[m])
            for k in base_mutable[m]:
                for i, l in enumerate(base_mutable[m][k]):
                    if x == 0:
                        agents = [{} for j in range(base_mutable[m][k].shape[0])]
                        x+=1
                    if m not in agents[i]:
                        agents[i][m] = {}
                    agents[i][m][k] = np.array([l])
    
    if save_to is None:
        path = os.getcwd()
    else:
        path = save_to

    for agent in agents:
        agent = hk.data_structures.to_immutable_dict(agent)


        name = names.get_full_name().replace(' ', '_') + '_' + agent_type +'.pkl' 

        with open(os.path.join(path, name), 'wb') as of:
            pickle.dump(agent, of)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Seperate in parallel trained agents")

    parser.add_argument(
        "--pkl_path", type=str, default=None,
        help="Path to weights of pretrained agents.")
    parser.add_argument(
        "--agent_type", type=str, default='rainbow',
        help="Agent specification")
    parser.add_argument(
        "--save_to", type=str, default=None,
        help="Path to save separated agents to")

    args = parser.parse_args()

    main(**vars(args))