# Unbuffered so log lines show up in `docker logs` immediately. The 2-minute timeout covers a panorama download plus
# inference; gunicorn's 30-second default would kill the worker and force every model to reload.
export PYTHONUNBUFFERED=1
# One log line per answered request, rejected ones included: who asked, when, what, the status, the response size and
# the time taken in ms. "Who" is the X-Forwarded-For header, since the https proxy is the only caller this server sees
# itself. Only the last address in it comes from the proxy; any before it were sent by the caller and can be made up.
ACCESS_LOG_FORMAT='%({x-forwarded-for}i)s %(t)s "%(r)s" %(s)s %(b)s %(M)sms'
# exec so gunicorn gets docker stop's signal and can finish requests in progress.
exec gunicorn -w 1 -b 0.0.0.0:5000 --timeout 120 --access-logfile - --access-logformat "$ACCESS_LOG_FORMAT" \
     "main:create_app()"
