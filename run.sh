clear
clear
clear
clear
clear

mkdir ./output
#clang++ ./src/*.c++ -fsanitize=address -Wall -std=c++17 -fsanitize=undefined -O3 -o ./output/out
clang++ ./src/*.c++ -Wall -std=c++17 -O3 -o ./output/out


if [[ $? -ne 0 ]]; then

    printf "\n\n[SHELL] Compilation failed\n\n"
    exit -1 

fi

printf "\n\n[SHELL] Compilation success\n\n"
./output/out

python3 ./src/plot.py

printf "\n\n[SHELL] Program exit code $?\n\n"



