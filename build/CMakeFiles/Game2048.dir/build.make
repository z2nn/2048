# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/aurora/2048

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/aurora/2048/build

# Include any dependencies generated for this target.
include CMakeFiles/Game2048.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/Game2048.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/Game2048.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Game2048.dir/flags.make

CMakeFiles/Game2048.dir/2048.cpp.o: CMakeFiles/Game2048.dir/flags.make
CMakeFiles/Game2048.dir/2048.cpp.o: ../2048.cpp
CMakeFiles/Game2048.dir/2048.cpp.o: CMakeFiles/Game2048.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/aurora/2048/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Game2048.dir/2048.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/Game2048.dir/2048.cpp.o -MF CMakeFiles/Game2048.dir/2048.cpp.o.d -o CMakeFiles/Game2048.dir/2048.cpp.o -c /home/aurora/2048/2048.cpp

CMakeFiles/Game2048.dir/2048.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Game2048.dir/2048.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/aurora/2048/2048.cpp > CMakeFiles/Game2048.dir/2048.cpp.i

CMakeFiles/Game2048.dir/2048.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Game2048.dir/2048.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/aurora/2048/2048.cpp -o CMakeFiles/Game2048.dir/2048.cpp.s

# Object files for target Game2048
Game2048_OBJECTS = \
"CMakeFiles/Game2048.dir/2048.cpp.o"

# External object files for target Game2048
Game2048_EXTERNAL_OBJECTS =

Game2048: CMakeFiles/Game2048.dir/2048.cpp.o
Game2048: CMakeFiles/Game2048.dir/build.make
Game2048: /usr/lib/x86_64-linux-gnu/libsfml-graphics.so
Game2048: /usr/lib/x86_64-linux-gnu/libsfml-window.so
Game2048: /usr/lib/x86_64-linux-gnu/libsfml-system.so
Game2048: CMakeFiles/Game2048.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/aurora/2048/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable Game2048"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Game2048.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Game2048.dir/build: Game2048
.PHONY : CMakeFiles/Game2048.dir/build

CMakeFiles/Game2048.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Game2048.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Game2048.dir/clean

CMakeFiles/Game2048.dir/depend:
	cd /home/aurora/2048/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/aurora/2048 /home/aurora/2048 /home/aurora/2048/build /home/aurora/2048/build /home/aurora/2048/build/CMakeFiles/Game2048.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Game2048.dir/depend
