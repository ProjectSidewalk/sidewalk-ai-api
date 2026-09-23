# Reads SIDEWALK_AI_API_KEY from the host's environment: export it before running this. It's the shared secret the
# Sidewalk webpage sends with each request, so it must match SIDEWALK_AI_API_KEY on every webpage instance.
# Bound to loopback so the key can only arrive over https: Apache on this host proxies to 127.0.0.1:60654 and
# nothing outside the machine can hit the plain-http port directly.
# Stop here rather than start a container that 401s everything (sudo drops the exported key, for one).
: "${SIDEWALK_AI_API_KEY:?export SIDEWALK_AI_API_KEY first}"
docker run   --gpus all   --runtime nvidia   -d   -p 127.0.0.1:60654:5000  --restart unless-stopped   -e SIDEWALK_AI_API_KEY   sidewalk-ai-api
