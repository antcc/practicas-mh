# Directorios del proyecto
SRC = src
INC = include
OBJ = obj
BIN = bin

# Opciones de compilación
CXX = g++
CXXFLAGS = -Wall -std=c++11 -g -I./$(INC) -O3

# Archivos del proyecto
SOURCES := util.cpp timer.cpp
INCLUDES := $(addprefix $(INC)/, $(SOURCES:.cpp=.h))
OBJECTS := $(addprefix $(OBJ)/, $(SOURCES:.cpp=.o))
SOURCES := $(addprefix $(SRC)/, $(SOURCES))

.PHONY: clean mrproper

all: p1 p2 p3
p1: $(BIN)/p1
p2: $(BIN)/p2
p3: $(BIN)/p3

# ************ Generación de ejecutables *************

# -- Práctica 3 --
$(BIN)/p3: $(OBJECTS) $(OBJ)/p3.o
	$(CXX) -o $@ $^

# -- Práctica 2 --
$(BIN)/p2: $(OBJECTS) $(OBJ)/p2.o
	$(CXX) -o $@ $^

# -- Práctica 1 --
$(BIN)/p1: $(OBJECTS) $(OBJ)/p1.o
	$(CXX) -o $@ $^

# ************ Compilación de módulos ************

# -- Prácticas --
$(OBJ)/%.o: $(SRC)/%.cpp $(INCLUDES)
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
