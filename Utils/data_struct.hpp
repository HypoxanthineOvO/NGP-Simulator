#include <vector>

template <typename T>
class FIFO {
public:
    FIFO(): fifoSize(2) {}
    FIFO(int size): fifoSize(size) {}

    bool write(T data) {
        if (fifo.size() < fifoSize) {
			Upd_Buffer = std::make_shared<T>(data);
            return true;
        }
		puts("FIFO is full!");
		exit(1);
        return false;
    }
    T read() {
        T data;
        if (fifo.size() > 0) {
            data = fifo[0];
            fifo.erase(fifo.begin());
            return data;
        }
		puts("FIFO is empty!");
		exit(1);
        return data;
    }
	void update() {
		if (Upd_Buffer != nullptr) {
			fifo.push_back(*Upd_Buffer);
			Upd_Buffer = nullptr;
		}
	}
    bool isFull() {
		full_check_cnt++;
		if (fifo.size() == fifoSize) full_cnt++;
        return fifo.size() == fifoSize;
    }
    bool isEmpty() {
		empty_check_cnt++;
		if (fifo.size() == 0) empty_cnt++;
        return fifo.size() == 0;
    }
    void printFIFO() {
        printf("Size: %d / %d\n", fifo.size(), fifoSize);
		printf("Full Check: %d / %d = %f\n", full_cnt, full_check_cnt, (float)full_cnt / full_check_cnt);
		printf("Empty Check: %d / %d = %f\n", empty_cnt, empty_check_cnt, (float)empty_cnt / empty_check_cnt);
	}

private:
    int fifoSize;
	std::shared_ptr<T> Upd_Buffer = nullptr;
    std::vector<T> fifo;

	// For Debug
	int full_check_cnt = 0, empty_check_cnt = 0;
	int full_cnt = 0, empty_cnt = 0;
};