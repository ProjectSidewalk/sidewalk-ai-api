# Reads SIDEWALK_AI_API_KEY from the host's environment: export it before running this. It's the shared secret the
# Sidewalk webpage sends with each request, so it must match SIDEWALK_AI_API_KEY on every webpage instance.
docker run   --gpus all   --runtime nvidia   -d   -p 60654:5000  --restart unless-stopped   -e SIDEWALK_AI_API_KEY   sidewalk-ai-api
