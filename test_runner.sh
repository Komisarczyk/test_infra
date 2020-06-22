#!/bin/sh
#needs to be in a directory containing polybench
FILES=$(ls -d */)
for FILE in $FILES 
    do 
        FILE=${FILE//\/}
        echo "$FILE"
        cd $FILE
        if [ "$FILE" == "stencils" ]; then
            KERNELS=$(ls -d */)
                for KERNEL in $KERNELS 
                    do 
                        KERNEL=${KERNEL//\/}
                        echo "$KERNEL"
                        cd $KERNEL
                        {
                            rm *.mlir
                            mlir-pet -I ../../../polybench/utilities -I ../../../polybench/linear-algebra/$FILE/gemver -I ../../../polybench/utilities/polybench.c ../../../polybench/$FILE/$KERNEL/$KERNEL.c > $KERNEL.mlir
                            make clean                         
                            make all
                        } &> /dev/null
                        #clang-format-3.9 -i main.c
                        SCRIPT=$(ls *.out)
                        {
                            ./$SCRIPT
                            echo $?
                        }
                        cd ..
                done
        else
            KERNELS=$(ls -d */)
                for KERNEL in $KERNELS 
                    do 
                        KERNEL=${KERNEL//\/}
                        echo "$KERNEL"
                        cd $KERNEL
                        {
                            rm *.mlir
                            mlir-pet -I ../../../polybench/utilities -I ../../../polybench/linear-algebra/$FILE/gemver -I ../../../polybench/utilities/polybench.c ../../../polybench/linear-algebra/$FILE/$KERNEL/$KERNEL.c > $KERNEL.mlir
                            make clean  
                            make all
                        } &> /dev/null
                        #clang-format-3.9 -i main.c
                        SCRIPT=$(ls *.out)
                        {
                            ./$SCRIPT
                            echo $?
                        }
                        cd ..
                done
        fi
        cd ..
done