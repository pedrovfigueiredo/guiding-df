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
	parser.add_argument("--method", type=str, default="reference", help="Method name")
	parser.add_argument("--bounce", type=int, default=6, help="Maximum path length")
	parser.add_argument("--budget_type", type=str, default="spp", help="Budget type (spp or time)")
	parser.add_argument("--budget_value", type=int, default=64000, help="Budget value")
	args = parser.parse_args()
	
	tmp_path = "tmp"
	os.makedirs(tmp_path, exist_ok=True)

	for config_scene_file in test_scenes:
		config_method = json.load(open("common/configs/render/guided/{}.json".format(args.method)))
		# load config
		config_method["passes"][0]["params"]["max_depth"] = args.bounce
		config_method["passes"][0]["params"]["mode"] = "offline"
		config_method["passes"][0]["params"]["budget"]["type"] = args.budget_type
		config_method["passes"][0]["params"]["budget"]["value"] = args.budget_value
		config_method["passes"][0]["params"]["bsdf_fraction"] = 1.0
		config_method["passes"][0]["params"]["auto_train"] = False
		
		# Remove the second pass if it exists
		if len(config_method["passes"]) > 1:
			del config_method["passes"][1]

		# Save config_method to a tmp file
		tmp_config_method = os.path.join(tmp_path, "{}.json".format(args.method))
		with open(tmp_config_method, "w") as f:
			json.dump(config_method, f, indent = 6)

		scene_config = config_scene_file[0]
		config_scene = json.load(open(scene_config))
		method_name = os.path.splitext(os.path.basename(args.method))[0].replace("/", "-")
		scene_name = os.path.splitext(os.path.basename(scene_config))[0]
		print("Testing scene: {}".format(scene_name))	
		config_scene["global"] = {
			"reference": "common/configs/references/{}/reference_{}b.exr".format(scene_name, args.bounce),
			"name": method_name
		}
		config_scene["output"] = "common/outputs/{}".format(scene_name)
		
		# Save config_scene to a tmp file
		tmp_config_scene = os.path.join(tmp_path, "{}.json".format(scene_name))
		with open(tmp_config_scene, "w") as f:
			json.dump(config_scene, f, indent = 6)

		output = subprocess.run([args.exe, "-method", tmp_config_method, "-scene", tmp_config_scene], capture_output=True)
		
		if output.returncode != 0:
			print("Error: {}".format(output.stderr))
			sys.exit(1)
		
		print(output.stdout)
		print("\n>>>>>>>>>>>>>Done rendering reference image for: {}".format(scene_name))
	
	print("Done rendering all scenes.\nDeleting tmp files...", end=" ")
	shutil.rmtree(tmp_path)
	print("Done.")