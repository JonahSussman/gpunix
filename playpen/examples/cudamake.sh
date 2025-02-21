filename="${1%.*}"  # Remove the file extension from the first argument
nvcc -g -G -arch=sm_61 --extended-lambda -dc "$1" -o "${filename}.o"
nvcc -g -G -arch=sm_61 --extended-lambda -dlink "${filename}.o" -o "${filename}_link.o"
nvcc -g -G -arch=sm_61 --extended-lambda "${filename}.o" "${filename}_link.o" -o "${filename}.out"