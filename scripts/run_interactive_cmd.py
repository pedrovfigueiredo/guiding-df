import os
import sys
import shutil
import json
import argparse
import subprocess

scene_config = "common/configs/scenes/salle-de-bain.json"


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--exe", type=str, default=".\\build\\src\\Release\\testbed.exe", help="Path to the executable")
	parser.add_argument("--method", type=str, default="guiding_df", help="Method name")
	parser.add_argument("--bounce", type=int, default=6, help="Maximum path length")
	parser.add_argument("--bsdf_fraction", type=float, default=0.3, help="BSDF/Guided sampling ratio")
	parser.add_argument("--experiment_name", type=str, default="", help="Experiment name")
	args = parser.parse_args()
	
	method = args.method
	
	tmp_path = "tmp"
	os.makedirs(tmp_path, exist_ok=True)

	config_method = json.load(open("common/configs/render/{}.json".format(method)))
	# load config
	config_method["passes"][0]["params"]["max_depth"] = args.bounce
	config_method["passes"][0]["params"]["bsdf_fraction"] = args.bsdf_fraction
	config_method["passes"][0]["params"]["mode"] = "interactive"
	config_method["passes"][0]["params"]["auto_train"] = False
	config_method["passes"][0]["params"]["budget"]["type"] = "none"
	
	
	# Save config_method to a tmp file
	tmp_config_method = os.path.join(tmp_path, "{}_{}_interactive.json".format(method, args.experiment_name))
	with open(tmp_config_method, "w") as f:
		json.dump(config_method, f, indent = 6)

	config_scene = json.load(open(scene_config))
	method_name = os.path.splitext(os.path.basename(method))[0].replace("/", "-")
	scene_name = os.path.splitext(os.path.basename(scene_config))[0]
	config_scene["global"] = {
		"name": method_name
	}
	config_scene["output"] = "common/outputs/{}".format(scene_name)
	
	# Save config_scene to a tmp file
	tmp_config_scene = os.path.join(tmp_path, "{}.json".format(scene_name))
	with open(tmp_config_scene, "w") as f:
		json.dump(config_scene, f, indent = 6)

	cmd = [args.exe, "-method", tmp_config_method, "-scene", tmp_config_scene]
	output = subprocess.run(cmd, capture_output=True)
	
	if output.returncode != 0:
		print("Error: {}".format(output.stderr))
		sys.exit(1)
	
	print(output.stdout)
	print("Done rendering.\nDeleting tmp files...", end=" ")
	shutil.rmtree(tmp_path)
	print("Done.")