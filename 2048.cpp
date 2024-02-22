#include <SFML/Graphics.hpp>
#include <iostream>
#include <cstdlib> 
#include <cmath>

const int SIZE = 4;

class Game2048 {
public:
    Game2048() {
        // 初始化游戏棋盘
        for (int i = 0; i < SIZE; ++i) {
            for (int j = 0; j < SIZE; ++j) {
                board[i][j] = 0;
            }
        }

        // 在两个随机位置生成初始数字方块
        generateRandomBlock();
        generateRandomBlock();
          // 加载字体，使用绝对路径
        if (!font.loadFromFile("../DejaVuSans-Bold.ttf")) {
            std::cerr << "Failed to load font file!" << std::endl;
        }
    }

    void draw(sf::RenderWindow& window) {
        // 清空窗口
        window.clear();

        // 绘制棋盘和数字方块
        for (int i = 0; i < SIZE; ++i) {
            for (int j = 0; j < SIZE; ++j) {
                drawBlock(window, i, j);
            }
        }

        // 显示窗口内容
        window.display();
    }

    void handleInput(sf::Event& event) {
    // 在每次按键前重置标志
    ifgenerate=false;
  // 处理方向键输入，更新游戏状态
    if (event.type == sf::Event::KeyPressed) {
        switch (event.key.code) {
            case sf::Keyboard::Up:
                moveUp();
                break;
            case sf::Keyboard::Down:
                moveDown();
                break;
            case sf::Keyboard::Left:
                moveLeft();
                break;
            case sf::Keyboard::Right:
                moveRight();
                break;
            case sf::Keyboard::R:
                resetGame();
                break; 
            default:
                break;
        }

        // 在每次方向键按下后生成一个新的数字方块
        if(ifgenerate){
            generateRandomBlock();
        }   
    }
    }

void endGame(sf::RenderWindow& window) {
 if (gameOver) {
        sf::Text gameOverText("Game Over", font, 60);
        sf::FloatRect textRect = gameOverText.getLocalBounds();
        gameOverText.setOrigin(textRect.left + textRect.width / 2.0f, textRect.top + textRect.height / 2.0f);
        gameOverText.setPosition(window.getSize().x / 2.0f, window.getSize().y / 2.0f);
        gameOverText.setFillColor(sf::Color::Red);
        window.draw(gameOverText);

        window.display();
        sf::sleep(sf::seconds(4)); 

        window.close();
    }
}

    void resetGame() {
        for (int i = 0; i < SIZE; ++i) {
            for (int j = 0; j < SIZE; ++j) {
                board[i][j] = 0;
            }
        }

        generateRandomBlock();
        generateRandomBlock();

        gameOver = false;
    }
private:
    int board[SIZE][SIZE];
    sf::Font font;  // 添加字体成员变量
    bool ifgenerate;
    bool gameOver = false;
    const sf::Color colors[12] = {
    sf::Color::White,
    sf::Color(255, 165, 0),  
    sf::Color(255, 69, 0),   
    sf::Color(255, 0, 0),    
    sf::Color(0, 255, 0),    
    sf::Color(0, 128, 255),  
    sf::Color(0, 255, 255),  
    sf::Color(255, 0, 255),  
    sf::Color(255, 255, 0),  
    sf::Color(255, 140, 0),  
    sf::Color(148, 0, 211),   
    sf::Color::Black
};
void drawBlock(sf::RenderWindow& window, int row, int col) {
        sf::RectangleShape block(sf::Vector2f(100.f, 100.f));
        block.setOutlineThickness(2.f);
        block.setOutlineColor(sf::Color::Black);
        block.setPosition(col * 100.f, row * 100.f);

        // 设置数字方块的颜色和文本
        int value = board[row][col];
        if (value >= 2 && value <= 2048) {
            block.setFillColor(colors[static_cast<int>(log2(value))]); // 使用log2将数字映射到颜色数组索引
        } else {
        block.setFillColor(sf::Color::White); // 默认颜色
    }

        window.draw(block);  // 先绘制方块

        // 绘制文字
        if (board[row][col] != 0) {
            int textSize;
            if(board[row][col]<100){
                textSize=60;
            }
            else if(board[row][col]<1000){
                textSize=40;
            }
            else{
                textSize=35;
            }
            sf::Text text(std::to_string(board[row][col]), font, textSize);

            
            // 调整绘制位置以居中显示
            sf::FloatRect textRect = text.getLocalBounds();
            text.setOrigin(textRect.left + textRect.width / 2.0f, textRect.top + textRect.height / 2.0f);
            text.setPosition(col * 100.f + 50.f, row * 100.f + 50.f);

            window.draw(text);
        }
    }

void moveUp() {
    for (int col = 0; col < SIZE; ++col) {
        for (int row = 1; row < SIZE; ++row) {
            if (board[row][col] != 0) {
                // 寻找上方可合并的方块
                int newRow = row - 1;
                while (newRow >= 0 && board[newRow][col] == 0) {
                    --newRow;
                }

                // 移动方块
                if (newRow + 1 != row) {
                    board[newRow + 1][col] = board[row][col];
                    board[row][col] = 0;
                    ifgenerate=true;
                }

                // 合并相同数字的方块
                if (newRow >= 0 && board[newRow][col] == board[newRow + 1][col]) {
                    board[newRow][col] *= 2;
                    board[newRow + 1][col] = 0;
                    ifgenerate=true;
                }
                    checkFor2048();
                    checkForGameOver();
            }
        }
    }
}

void moveDown() {
    for (int col = 0; col < SIZE; ++col) {
        for (int row = SIZE - 2; row >= 0; --row) {
            if (board[row][col] != 0) {
                // 寻找下方可合并的方块
                int newRow = row + 1;
                while (newRow < SIZE && board[newRow][col] == 0) {
                    ++newRow;
                }

                // 移动方块
                if (newRow - 1 != row) {
                    board[newRow - 1][col] = board[row][col];
                    board[row][col] = 0;
                    ifgenerate=true;
                }

                // 合并相同数字的方块
                if (newRow < SIZE && board[newRow][col] == board[newRow - 1][col]) {
                    board[newRow][col] *= 2;
                    board[newRow - 1][col] = 0;
                    ifgenerate=true;
                }
                    checkFor2048();
                    checkForGameOver();
            }
        }
    }
}

void moveLeft() {
    for (int row = 0; row < SIZE; ++row) {
        for (int col = 1; col < SIZE; ++col) {
            if (board[row][col] != 0) {
                // 寻找左方可合并的方块
                int newCol = col - 1;
                while (newCol >= 0 && board[row][newCol] == 0) {
                    --newCol;
                }

                // 移动方块
                if (newCol + 1 != col) {
                    board[row][newCol + 1] = board[row][col];
                    board[row][col] = 0;
                    ifgenerate=true;
                }

                // 合并相同数字的方块
                if (newCol >= 0 && board[row][newCol] == board[row][newCol + 1]) {
                    board[row][newCol] *= 2;
                    board[row][newCol + 1] = 0;
                    ifgenerate=true;
                }
                    checkFor2048();
                    checkForGameOver();
            }
        }
    }
}

void moveRight() {
    for (int row = 0; row < SIZE; ++row) {
        for (int col = SIZE - 2; col >= 0; --col) {
            if (board[row][col] != 0) {
                // 寻找右方可合并的方块
                int newCol = col + 1;
                while (newCol < SIZE && board[row][newCol] == 0) {
                    ++newCol;
                }

                // 移动方块
                if (newCol - 1 != col) {
                    board[row][newCol - 1] = board[row][col];
                    board[row][col] = 0;
                    ifgenerate=true;
                }

                // 合并相同数字的方块
                if (newCol < SIZE && board[row][newCol] == board[row][newCol - 1]) {
                    board[row][newCol] *= 2;
                    board[row][newCol-1] = 0;
                    ifgenerate=true;
                }
                    checkFor2048();
                    checkForGameOver();
            }
        }
    }
}

void checkFor2048() {
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            if (board[i][j] == 2048) {
                gameOver = true;
            }
        }
    }
}


    void checkForGameOver() {
        // 检查是否有空位
        for (int i = 0; i < SIZE; ++i) {
            for (int j = 0; j < SIZE; ++j) {
                if (board[i][j] == 0) {
                    return; 
                }
            }
        }
        // 检查上下左右是否有相同数字
        for (int i = 0; i < SIZE; ++i) {
            for (int j = 0; j < SIZE; ++j) {
                int currentValue = board[i][j];

                if (i > 0 && board[i - 1][j] == currentValue) {
                    return; 
                }

                if (i < SIZE - 1 && board[i + 1][j] == currentValue) {
                    return; 
                }

                if (j > 0 && board[i][j - 1] == currentValue) {
                    return; 
                }

                if (j < SIZE - 1 && board[i][j + 1] == currentValue) {
                    return;  
                }
            }
        }

        gameOver = true;
    }

   void generateRandomBlock() {
        // 在一个随机空位置生成一个数字方块（2或4）
        int row, col;
        do {
            row = rand() % SIZE;
            col = rand() % SIZE;
        } while (board[row][col] != 0);

        // 80%的概率生成2，20%的概率生成4
        int value = (rand() % 5 == 0) ? 4 : 2;
        board[row][col] = value;
    }
};

int main() {
    sf::RenderWindow window(sf::VideoMode(400, 400), "2048 Game");

    Game2048 game;

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }

            // 处理用户输入
            game.handleInput(event);
        }

        // 更新并绘制游戏状态
        game.draw(window);
        game.endGame(window);
    }

    return 0;
}