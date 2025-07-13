import os
import sys
import shutil
import json
import argparse
import subprocess

test_scenes = [
	["common/configs/scenes/bathroom.json"],
	["common/configs/scenes/bedroom.json"],
	["common/configs/scenes/breakfast.json"],
	["common/configs/scenes/kitchen.json"],
	["common/configs/scenes/salle-de-bain.json"],
	["common/configs/scenes/staircase.json"],
	["common/configs/scenes/veach-ajar.json"],
]


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--exe", type=str, default=".\\build\\src\\Release\\testbed.exe", help="Path to the executable")
	parser.add_argument("--method", type=str, default="guiding_df", help="Method name")
	parser.add_argument("--bounce", type=int, default=6, help="Maximum path length")
	parser.add_argument("--bsdf_fraction", type=float, default=0.3, help="BSDF/Guided sampling ratio")
	parser.add_argument("--train_budget_spp", type=float, default=0.33, help="Budget percentage for training")
	parser.add_argument("--train_budget_time", type=float, default=0.3, help="Budget percentage for training")
	parser.add_argument("--budget_type", type=str, default="spp", help="Budget type (spp or time)")
	parser.add_argument("--budget_value", type=int, default=1000, help="Budget value")
	parser.add_argument("--n_runs", type=int, default=1, help="Number of runs for each scene")
	parser.add_argument("--experiment_name", type=str, default="", help="Experiment name")
	parser.add_argument("--image_save_pass_interval", type=int, default=-1, help="Save image every n passes. Same unit as budget value.")
	args = parser.parse_args()

	methods = [m.strip() for m in args.method.split(",") if m.strip()]
	
	tmp_path = "tmp"
	os.makedirs(tmp_path, exist_ok=True)

	for method in methods:
		for config_scene_file in test_scenes:
			config_method = json.load(open("common/configs/render/{}.json".format(method)))
			# load config
			config_method["passes"][0]["params"]["max_depth"] = args.bounce
			config_method["passes"][0]["params"]["bsdf_fraction"] = args.bsdf_fraction
			config_method["passes"][0]["params"]["mode"] = "offline"
			config_method["passes"][0]["params"]["auto_train"] = True
			config_method["passes"][0]["params"]["training_budget_spp"] = args.train_budget_spp
			config_method["passes"][0]["params"]["training_budget_time"] = args.train_budget_time
			config_method["passes"][0]["params"]["budget"]["type"] = args.budget_type
			config_method["passes"][0]["params"]["budget"]["value"] = args.budget_value
			
			if args.image_save_pass_interval > 0:
				config_method["passes"].append({
					"enable": True,
					"name": "ImageSavePass",
					"params": {
						"continuous": True,
						"interval_type": args.budget_type,
						"interval": args.image_save_pass_interval,
						"log": False,
						"save_at_finalize": True,
						"format": "hdr"
					}
				})
			
			for run_i in range(args.n_runs):
				# Save config_method to a tmp file
				tmp_config_method = os.path.join(tmp_path, "{}_{}_{}.json".format(method, args.experiment_name, run_i))
				with open(tmp_config_method, "w") as f:
					json.dump(config_method, f, indent = 6)

				scene_config = config_scene_file[0]
				config_scene = json.load(open(scene_config))
				method_name = os.path.splitext(os.path.basename(method))[0].replace("/", "-")
				scene_name = os.path.splitext(os.path.basename(scene_config))[0]
				print("Testing scene: {} at run {}".format(scene_name, run_i))	
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
				print("\n>>>>>>>>>>>>>Done rendering image for: {}-{}".format(scene_name, run_i))
		
	print("Done rendering all scenes.\nDeleting tmp files...", end=" ")
	shutil.rmtree(tmp_path)
	print("Done.")