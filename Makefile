all:
	python src/main.py

clean:
	rm -f *.o

fclean: clean

re: fclean all

.PHONY: clean fclean all re
