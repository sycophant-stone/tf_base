#!/bin/bash
bash_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo $bash_dir
echo ${BASH_SOURCE}
echo ${BASH_SOURCE[0]}

