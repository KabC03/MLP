clear

mkdir ./output
python3 ./src/generate_tokeniser_enum.py
clang++ ./src/*.c++ -fsanitize=address -Wall -Werror -std=c++11 -O0 -o ./output/out

if [[ $? -ne 0 ]]; then

    printf "\n\n[SHELL] Compilation failed\n\n"
    exit -1 

fi

printf "\n\n[SHELL] Compilation success\n\n"
./output/out


printf "\n\n[SHELL] Program exit code $?\n\n"



