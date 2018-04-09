#set -e -u

trap "kill 0" EXIT

function gen_trace()
{
	SPORT=$(python3 -c "
import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('', 0))
addr = s.getsockname()
print(addr[1])
s.close()
")

	VPORT=$(python3 -c "
import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('', 0))
addr = s.getsockname()
print(addr[1])
s.close()
")

	TEMP=$(mktemp)
	pushd ../.. > /dev/null
	export -p > $TEMP
	source venv/bin/activate
	echo "Starting server (ports: $SPORT & $VPORT)..."
	#(echo start && sleep 10000) | python3 serveur.py --sport $SPORT --vport $VPORT > /dev/null 2> /dev/null &
	(echo start && sleep 10000) | python3 serveur.py --map static/assets/tilemaps/maps/arbre.json --sport $SPORT --vport $VPORT > /dev/null 2> /dev/null &
	PID_SERVER=$!
	source $TEMP
	popd > /dev/null
	rm $TEMP

	sleep 0.5
	echo "Starting client..."
	python3 main.py --mode play --port $SPORT > /dev/null &
	PID_CLIENT=$!

	echo "Waiting client..."
	wait $PID_CLIENT
	echo "Killing server..."
    kill -9 $PID_SERVER
	echo "Done"
}

for (( j=0 ; j<10 ; ++j ))
{
	for (( i=0 ; i<10 ; ++i ))
	{
		gen_trace &
		#sleep 4
		#gen_trace &
		wait
	}

	wait
}
