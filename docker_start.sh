# Export SIDEWALK_AI_API_KEY (the password the Sidewalk websites send) before running this; it's passed through to the
# container. The port is bound to 127.0.0.1 so the only way in is through the https proxy on this machine.
# Refuse to start without a password (note that sudo drops exported variables).
: "${SIDEWALK_AI_API_KEY:?export SIDEWALK_AI_API_KEY first}"
docker run   --gpus all   --runtime nvidia   -d   -p 127.0.0.1:60654:5000  --restart unless-stopped   -e SIDEWALK_AI_API_KEY   sidewalk-ai-api
