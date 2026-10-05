/* A foreign-architecture syscall from an x86_64 process (#1503).
 *
 * `int $0x80` enters the kernel through the i386 ABI even in a 64-bit
 * process, where syscall numbers mean different things.  #1503's filter
 * checks the architecture first and KILLS the process for a foreign one
 * (SIGSYS), so under the filter this program dies; without it, it prints
 * the pid that i386 getpid (20) returned and exits 0.
 *
 *   gcc -O0 -o int80 int80.c
 *   ./int80; echo "exit=$?"     # 159 (= 128 + SIGSYS 31) under the filter
 */
#include <stdio.h>

int main(void) {
    long r;
    __asm__ volatile ("int $0x80" : "=a"(r) : "a"(20L) : "memory");
    printf("i386 getpid via int 0x80 returned %ld\n", r);
    return 0;
}
