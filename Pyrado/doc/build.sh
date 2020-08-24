sphinx-apidoc -o source -t source/templates ..
cd source
for file in pyrado.*.rst; do
  if [ -e "$file" ]; then
    newname=`echo "$file" | sed 's/^pyrado.\(.*\)\.rst$/\1.rst/'`
    mv "$file" "$newname"
  fi
done
cd ..
sphinx-build -b html source build