from medacy.pipeline_components import MetaMap

metamap = MetaMap(metamap_path="/home/share/programs/metamap/2016/public_mm/bin/metamap", cache_output=True)


file_to_map = "/home/aymulyar/development/medaCy/medacy/tests/pipeline_components/metamap/test.txt"
file = metamap.map_file(file_to_map)
print(file)
