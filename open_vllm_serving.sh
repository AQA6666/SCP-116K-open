export CUDA_VISIBLE_DEVICES=0,1,2,3
export PORT_NUM=8998
export MODEL_PATH=/ddn/gemini/gemini-sharedata/space/ros2p9tmh1z4/Models/Qwen/Qwen3-Next-80B-A3B-Instruct
export MODEL_NAME=Qwen3-Next-80B-A3B-Instruct
export TP=4

# Check if required variables are provided via environment variables
if [ -z "${PORT_NUM}" ]; then
    echo "Error: PORT_NUM environment variable is not set"
    exit 1
fi

if [ -z "${MODEL_NAME}" ]; then
    echo "Error: MODEL_NAME environment variable is not set"
    exit 1
fi

if [ -z "${MODEL_PATH}" ]; then
    echo "Error: MODEL_PATH environment variable is not set"
    exit 1
fi

if [ -z "${TP}" ]; then
    echo "Error: TP environment variable is not set"
    exit 1
fi

set -ex

TIMESTAMP=$(date +%Y%m%d%H%M%S)

python -m vllm.entrypoints.openai.api_server \
    --port ${PORT_NUM} \
    --model ${MODEL_PATH} \
    --served-model-name ${MODEL_NAME} \
    --compilation_config.cudagraph_mode=PIECEWISE \
    --max-model-len 32768 \
    -tp ${TP} > /ddn/shaojw_group/ludakuan/log/vllm_serving_${MODEL_NAME}_${TIMESTAMP}.out &

SERVICE_PID=$!
SERVICE_PORT=$PORT_NUM
echo "Service starting with PID $SERVICE_PID..."

# apt-get install net-tools
is_port_in_use() {
    # netstat -an | grep "$SERVICE_PORT" | grep LISTEN #
    netstat -an | grep "[.:]$1\b" | grep LISTEN
    return $?
}

while ! is_port_in_use $SERVICE_PORT; do
# while true; do
    echo "Waiting for service to start on port $SERVICE_PORT..."
    sleep 10
done

echo "Service is now listening on port $SERVICE_PORT." 