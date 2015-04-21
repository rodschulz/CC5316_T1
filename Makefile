IS_DEBUG	= y

# compiler and linker flags
CXXFLAGS	= -Wall -Wextra -pedantic -Wno-unused-function -fmessage-length=0 -std=gnu++11 
LDFLAGS		= -lpthread -lm

# enable/disable debug
ifeq ($(IS_DEBUG),y)
	CXXFLAGS += -O0 -ggdb
else
	CXXFLAGS += -O3
endif

# package configuration to use openCV
ifneq ($(shell pkg-config --exists opencv && echo ok),ok)
	$(warning pkg-config can't find required library "opencv")
endif

CXXFLAGS	+= $(shell pkg-config --cflags opencv)
LDFLAGS		+= $(shell pkg-config --libs   opencv)

# folders and files locations
BUILD_DIR	= build
SOURCE_DIR	= src
TARGET		= $(BUILD_DIR)/Homework1
OBJS		= $(BUILD_DIR)/Homework1.o $(BUILD_DIR)/Helper.o $(BUILD_DIR)/Printer.o $(BUILD_DIR)/Loader.o

# targets and rules
.PHONY: all clean

$(BUILD_DIR)/%.o: $(SOURCE_DIR)/%.cpp $(SOURCE_DIR)/%.h
	mkdir -p $(BUILD_DIR)
	$(CXX) -o $@ -c $< $(CXXFLAGS)

$(BUILD_DIR)/%.o: $(SOURCE_DIR)/%.cpp
	mkdir -p $(BUILD_DIR)
	$(CXX) -o $@ -c $< $(CXXFLAGS)

$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS) 

all: $(TARGET)

clean:
	rm -rf $(BUILD_DIR)/*

