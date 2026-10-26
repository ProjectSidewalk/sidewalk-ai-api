# Unbuffered so log lines show up in `docker logs` immediately. The 2-minute timeout covers a panorama download plus
# inference; gunicorn's 30-second default would kill the worker and force every model to reload.
export PYTHONUNBUFFERED=1
# exec so gunicorn gets docker stop's signal and can finish requests in progress.
exec gunicorn -w 1 -b 0.0.0.0:5000 --timeout 120 "main:create_app()"
