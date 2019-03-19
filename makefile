# Directorios del proyecto
SRC = src
INC = include
OBJ = obj
BIN = bin

# Opciones de compilación
CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++11 -g -I./$(INC)

# Archivos del proyecto
SOURCES := random.cpp util.cpp
INCLUDES := $(addprefix $(INC)/, $(SOURCES:.cpp=.h))
OBJECTS := $(addprefix $(OBJ)/, $(SOURCES:.cpp=.o))
SOURCES := $(addprefix $(SRC)/, $(SOURCES))

.PHONY: clean mrproper

# ************ Generación de ejecutables *************

# -- Práctica 1 --
$(BIN)/p1: $(OBJECTS) $(OBJ)/p1.o
	$(CXX) -o $@ $^

# ************ Compilación de módulos ************

# -- Práctica 1 --
$(OBJ)/p1.o: $(SRC)/p1.cpp $(INCLUDES)
	$(CXX) $(CXXFLAGS) -o $@ -c $<

# -- Utilidades --
$(OBJ)/%.o: $(SRC)/%.cpp $(INC)/%.h
	$(CXX) $(CXXFLAGS) -o $@ -c $<

# ************ Limpieza ************
clean :
	-@rm -f $(OBJ)/* $(SRC)/*~ $(INC)/*~ *~
	@echo "Limpiando..."

mrproper : clean
	-@rm -f $(BIN)/*
	@echo "Limpieza completada"
