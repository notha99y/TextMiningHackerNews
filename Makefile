.PHONY : help
help : Makefile
	@ sed -n 's/^##//p' $<

## create_mongodb: Create MongoDB given a json file placed in data/raw/*.json
create_mongodb :
	python src/make_mongodb.py
