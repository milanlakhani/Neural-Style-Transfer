parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
content_directory="$parent_path/data/content-images"
style_directory="$parent_path/data/style-images"
output_directory="$parent_path/data/output-images"

for content_file in $(ls $content_directory); do
    echo "Content File: $content_file"
    export "CONTENT_IMAGE=$content_file"
    for style_file in $(ls $style_directory); do
        if [ "$style_file" = "$content_file" ]; then
            # Code to execute if condition is true
            echo "Style and Content are the same"
                continue
        fi
        echo "Style File: $style_file"
        export "STYLE_IMAGE=$style_file"
        if [[ ! " $(ls $output_directory) " =~ " combined_${content_file%%.*}_${style_file%%.*} " ]]; then
        	python3 NST.py
        else
        	echo "Combination already in output directory"
        fi
    done
done
