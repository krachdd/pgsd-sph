// Copyright (c) 2016-2023 The Regents of the University of Michigan
// Part of GSD, released under the BSD 2-Clause License.

// #ifdef ENABLE_MPI

#include "pgsd.h"

#include <sys/stat.h>
#ifdef _WIN32

#pragma warning(push)
#pragma warning(disable : 4996)


#define PGSD_USE_MMAP 0
#define WIN32_LEAN_AND_MEAN
#include <io.h>
#include <windows.h>

#else // linux / mac

#define _XOPEN_SOURCE 500
#include <sys/mman.h>
#include <unistd.h>
// #define PGSD_USE_MMAP 1
#define PGSD_USE_MMAP 0
#define PGSD_ACTIVATE_LOGGER 0

// for sys/mman.h and mmap see:
// https://pubs.opengroup.org/onlinepubs/000095399/functions/mmap.html

#endif

#ifdef __APPLE__
#include <limits.h>
#endif

#include <errno.h>
#include <fcntl.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

// Only testing
#include <inttypes.h>


/** @file pgsd.c
    @brief Implements the PGSD C API
*/

/// Magic value identifying a PGSD file
const uint64_t PGSD_MAGIC_ID = 0x65DF65DF65DF65DF;

/// Initial index size
enum
    {
    PGSD_INITIAL_INDEX_SIZE = 128
    };

/// Initial namelist size
enum
    {
    PGSD_INITIAL_NAME_BUFFER_SIZE = 1024
    };

/// Size of initial frame index
enum
    {
    PGSD_INITIAL_FRAME_INDEX_SIZE = 16
    };

/// Initial size of write buffer
enum
    {
    PGSD_INITIAL_WRITE_BUFFER_SIZE = 1024
    };

/// Default maximum size of write buffer
enum
    {
    PGSD_DEFAULT_MAXIMUM_WRITE_BUFFER_SIZE = 64 * 1024 * 1024
    };

/// Default number of index entries to buffer
enum
    {
    PGSD_DEFAULT_INDEX_ENTRIES_TO_BUFFER = 256 * 1024
    };

/// Size of hash map
enum
    {
    PGSD_NAME_MAP_SIZE = 57557
    };

/// Current PGSD file specification
enum
    {
    PGSD_CURRENT_FILE_VERSION = 2
    };

// Helper functions to regarding the parallelization

void bcast_file_size(struct pgsd_handle* handle)
    {
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&handle->file_size, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    }

void bcast_frame_index_size(struct pgsd_handle* handle)
    {
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&handle->frame_index.size, 1, my_MPI_SIZE_T, 0, MPI_COMM_WORLD);
    }

void bcast_index_allocated_entries(struct pgsd_handle* handle)
    {
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&handle->header.index_allocated_entries, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    }

void bcast_index_location(struct pgsd_handle* handle)
    {
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&handle->header.index_location, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    }

void bcast_namelist_allocated_entries(struct pgsd_handle* handle)
    {
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&handle->header.namelist_allocated_entries, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    }

void bcast_namelist_location(struct pgsd_handle* handle)
    {
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&handle->header.namelist_location, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    }

void bcast_number_of_names_file(struct pgsd_handle* handle)
    {
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&handle->file_names.n_names, 1, my_MPI_SIZE_T, 0, MPI_COMM_WORLD);
    }

void bcast_number_of_names_frame(struct pgsd_handle* handle)
    {
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&handle->frame_names.n_names, 1, my_MPI_SIZE_T, 0, MPI_COMM_WORLD);
    }

void bcast_retval(int* retval)
    {
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(retval, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

// Check before use
void pgsd_bcast_index_entry(struct pgsd_index_entry* e)
    {
    int rank;
    const int f = 5;
    const int blocklegth[] = { 2, 1, 1, 1, 2 };
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Datatype entry;
    MPI_Datatype types[] = { MPI_UINT64_T,
                             MPI_INT64_T,
                             MPI_UINT32_T,
                             MPI_UINT16_T,
                             MPI_UINT8_T};
    MPI_Aint offsets[5];
    offsets[0] = offsetof(struct pgsd_index_entry, frame);
    offsets[1] = offsetof(struct pgsd_index_entry, location);
    offsets[2] = offsetof(struct pgsd_index_entry, M);
    offsets[3] = offsetof(struct pgsd_index_entry, id);
    offsets[4] = offsetof(struct pgsd_index_entry, type);
    MPI_Type_create_struct(f, blocklegth, offsets, types, &entry);
    MPI_Type_commit(&entry);
    MPI_Bcast(e, 1, entry, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Type_free(&entry);
    }

bool file_name_size_is_same(size_t s)
    {
    size_t p[2];
    p[0] = s;
    p[1] = -s;
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, p, 2, my_MPI_SIZE_T, MPI_MIN, MPI_COMM_WORLD);
    return ( p[0] == -p[1] );
    }

bool frame_is_same(uint64_t s)
    {
    uint64_t p[2];
    p[0] = s;
    p[1] = -s;
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, p, 2, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD);
    return ( p[0] == -p[1] );
    }

bool file_size_is_same(long long int s)
    {
    long long int p[2];
    p[0] = s;
    p[1] = -s;
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, p, 2, MPI_LONG_LONG_INT, MPI_MIN, MPI_COMM_WORLD);
    return ( p[0] == -p[1] );
    }



/** Zero memory

    @param d pointer to memory region
    @param size_to_zero size of the area to zero in bytes
*/
inline static void pgsd_util_zero_memory(void* d, size_t size_to_zero)
    {
    memset(d, 0, size_to_zero);
    }

/** @internal
    @brief Allocate a name/id map

    @param map Map to allocate.
    @param size Number of entries in the map.

    @returns PGSD_SUCCESS on success, PGSD_* error codes on error.
*/
inline static int pgsd_name_id_map_allocate(struct pgsd_name_id_map* map, size_t size)
    {
    if (map == NULL || map->v || size == 0 || map->size != 0)
        {
        return PGSD_ERROR_INVALID_ARGUMENT;
        }

    map->v = calloc(size, sizeof(struct pgsd_name_id_pair));
    if (map->v == NULL)
        {
        return PGSD_ERROR_MEMORY_ALLOCATION_FAILED;
        }

    map->size = size;

    return PGSD_SUCCESS;
    }

/** @internal
    @brief Free a name/id map

    @param map Map to free.

    @returns PGSD_SUCCESS on success, PGSD_* error codes on error.
*/
inline static int pgsd_name_id_map_free(struct pgsd_name_id_map* map)
    {
    if (map == NULL || map->v == NULL || map->size == 0)
        {
        return PGSD_ERROR_INVALID_ARGUMENT;
        }

    // free all of the linked lists
    size_t i;
    for (i = 0; i < map->size; i++)
        {
        free(map->v[i].name);

        struct pgsd_name_id_pair* cur = map->v[i].next;
        while (cur != NULL)
            {
            struct pgsd_name_id_pair* prev = cur;
            cur = cur->next;
            free(prev->name);
            free(prev);
            }
        }

    // free the main map
    free(map->v);

    map->v = 0;
    map->size = 0;

    return PGSD_SUCCESS;
    }

/** @internal
    @brief Hash a string

    @param str String to hash

    @returns Hashed value of the string.
*/
inline static unsigned long pgsd_hash_str(const unsigned char* str)
    {
    unsigned long hash = 5381; // NOLINT
    int c;

    while ((c = *str++))
        {
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c NOLINT */
        }

    return hash;
    }

/** @internal
    @brief Insert a string into a name/id map

    @param map Map to insert into.
    @param str String to insert.
    @param id ID to associate with the string.

    @returns PGSD_SUCCESS on success, PGSD_* error codes on error.
*/
inline static int pgsd_name_id_map_insert(struct pgsd_name_id_map* map, const char* str, uint16_t id)
    {
#if PGSD_ACTIVATE_LOGGER
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("[INFO]: Rank: %i -> PGSD: pgsd_name_id_map_insert\n", rank);
#endif
    if (map == NULL || map->v == NULL || map->size == 0)
        {
        return PGSD_ERROR_INVALID_ARGUMENT;
        }

    size_t hash = pgsd_hash_str((const unsigned char*)str) % map->size;

    // base case: no conflict
    if (map->v[hash].name == NULL)
        {
        map->v[hash].name = calloc(strlen(str) + 1, sizeof(char));
        if (map->v[hash].name == NULL)
            {
            return PGSD_ERROR_MEMORY_ALLOCATION_FAILED;
            }
        memcpy(map->v[hash].name, str, strlen(str) + 1);
        map->v[hash].id = id;
        map->v[hash].next = NULL;
        }
    else
        {
        // go to the end of the conflict list
        struct pgsd_name_id_pair* insert_point = map->v + hash;

        while (insert_point->next != NULL)
            {
            insert_point = insert_point->next;
            }

        // allocate and insert a new entry
        insert_point->next = malloc(sizeof(struct pgsd_name_id_pair));
        if (insert_point->next == NULL)
            {
            return PGSD_ERROR_MEMORY_ALLOCATION_FAILED;
            }

        insert_point->next->name = calloc(strlen(str) + 1, sizeof(char));
        if (insert_point->next->name == NULL)
            {
            return PGSD_ERROR_MEMORY_ALLOCATION_FAILED;
            }
        memcpy(insert_point->next->name, str, strlen(str) + 1);
        insert_point->next->id = id;
        insert_point->next->next = NULL;
        }

    return PGSD_SUCCESS;
    }

/** @internal
    @brief Find an ID in a name/id mapping

    @param map Map to search.
    @param str String to search.

    @returns The ID if found, or UINT16_MAX if not found.
*/
inline static uint16_t pgsd_name_id_map_find(struct pgsd_name_id_map* map, const char* str)
    {
    if (map == NULL || map->v == NULL || map->size == 0)
        {
        return UINT16_MAX;
        }

    size_t hash = pgsd_hash_str((const unsigned char*)str) % map->size;

    struct pgsd_name_id_pair* cur = map->v + hash;

    while (cur != NULL)
        {
        if (cur->name == NULL)
            {
            // not found
            return UINT16_MAX;
            }

        if (strcmp(str, cur->name) == 0)
            {
            // found
            return cur->id;
            }

        // keep looking
        cur = cur->next;
        }

    // not found in any conflict
    return UINT16_MAX;
    }

/** @internal
    @brief Utility function to validate index entry
    @param handle handle to the open pgsd file
    @param idx index of entry to validate

    @returns 1 if the entry is valid, 0 if it is not
*/
inline static int pgsd_is_entry_valid(struct pgsd_handle* handle, size_t idx)
    {
    const struct pgsd_index_entry entry = handle->file_index.data[idx];

    // check for valid type
    if (pgsd_sizeof_type((enum pgsd_type)entry.type) == 0)
        {
        return 0;
        }

    // validate that we don't read past the end of the file
    size_t size = entry.N * entry.M * pgsd_sizeof_type((enum pgsd_type)entry.type);
    if ((entry.location + size) > (uint64_t)handle->file_size)
        {
        return 0;
        }

    // check for valid frame (frame cannot be more than the number of index entries)
    if (entry.frame >= handle->header.index_allocated_entries)
        {
        return 0;
        }

    // check for valid id
    if (entry.id >= (handle->file_names.n_names + handle->frame_names.n_names))
        {
        return 0;
        }

    // check for valid flags
    if (entry.flags != 0)
        {
        return 0;
        }

    return 1;
    }

/** @internal
    @brief Allocate a write buffer

    @param buf Buffer to allocate.
    @param reserve Number of bytes to allocate.

    @returns PGSD_SUCCESS on success, PGSD_* error codes on error.
*/
inline static int pgsd_byte_buffer_allocate(struct pgsd_byte_buffer* buf, size_t reserve)
    {
    if (buf == NULL || buf->data || reserve == 0 || buf->reserved != 0 || buf->size != 0)
        {
        return PGSD_ERROR_INVALID_ARGUMENT;
        }

    buf->data = calloc(reserve, sizeof(char));
    if (buf->data == NULL)
        {
        return PGSD_ERROR_MEMORY_ALLOCATION_FAILED;
        }

    buf->size = 0;
    buf->reserved = reserve;

    return PGSD_SUCCESS;
    }

/** @internal
    @brief Append bytes to a byte buffer

    @param buf Buffer to append to.
    @param data Data to append.
    @param size Number of bytes in *data*.

    @returns PGSD_SUCCESS on success, PGSD_* error codes on error.

    DK : per rank
*/
inline static int pgsd_byte_buffer_append(struct pgsd_byte_buffer* buf, const char* data, size_t size)
    {
    if (buf == NULL || buf->data == NULL || size == 0 || buf->reserved == 0)
        {
        return PGSD_ERROR_INVALID_ARGUMENT;
        }

    if (buf->size + size > buf->reserved)
        {
        // reallocate by doubling
        size_t new_reserved = buf->reserved * 2;
        while (buf->size + size >= new_reserved)
            {
            new_reserved = new_reserved * 2;
            }

        char* old_data = buf->data;
        buf->data = realloc(buf->data, sizeof(char) * new_reserved);
        if (buf->data == NULL)
            {
            // this free should not be necessary, but clang-tidy disagrees
            free(old_data);
            return PGSD_ERROR_MEMORY_ALLOCATION_FAILED;
            }

        // zero the new memory, but only the portion after the end of the new section to be appended
        pgsd_util_zero_memory(buf->data + (buf->size + size),
                             sizeof(char) * (new_reserved - (buf->size + size)));
        buf->reserved = new_reserved;
        }

    memcpy(buf->data + buf->size, data, size);
    buf->size += size;

    return PGSD_SUCCESS;
    }

/** @internal
    @brief Free the memory allocated by the write buffer or unmap the mapped memory.

    @param buf Buffer to free.

    @returns PGSD_SUCCESS on success, PGSD_* error codes on error.

    DK : per rank

*/
inline static int pgsd_byte_buffer_free(struct pgsd_byte_buffer* buf)
    {
    if (buf == NULL || buf->data == NULL)
        {
        return PGSD_ERROR_INVALID_ARGUMENT;
        }

    free(buf->data);

    pgsd_util_zero_memory(buf, sizeof(struct pgsd_byte_buffer));
    return PGSD_SUCCESS;
    }

/** @internal
    @brief Allocate a buffer of index entries

    @param buf Buffer to allocate.
    @param reserve Number of entries to allocate.

    @post The buffer's data element has *reserve* elements allocated in memory.

    @returns PGSD_SUCCESS on success, PGSD_* error codes on error.

    DK : only used in root, all index buffers only used in root
*/
inline static int pgsd_index_buffer_allocate(struct pgsd_index_buffer* buf, size_t reserve)
    {
#if PGSD_ACTIVATE_LOGGER
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("[INFO]: Rank: %i -> PGSD: pgsd_index_buffer_allocate\n", rank);
#endif
    if (buf == NULL || buf->mapped_data || buf->data || reserve == 0 || buf->reserved != 0
        || buf->size != 0)
        {
        return PGSD_ERROR_INVALID_ARGUMENT;
        }

    buf->data = calloc(reserve, sizeof(struct pgsd_index_entry));
    if (buf->data == NULL)
        {
        return PGSD_ERROR_MEMORY_ALLOCATION_FAILED;
        }

    buf->size = 0;
    buf->reserved = reserve;
    buf->mapped_data = NULL;
    buf->mapped_len = 0;

    return PGSD_SUCCESS;
    }

/** @internal
    @brief Map index entries from the file

    @param buf Buffer to map.
    @param handle PGSD file handle to map.

    @post The buffer's data element contains the index data from the file.

    On some systems, this will use mmap to efficiently access the file. On others, it may result in
    an allocation and read of the entire index from the file.

    @returns PGSD_SUCCESS on success, PGSD_* error codes on error.
*/
inline static int pgsd_index_buffer_map(struct pgsd_index_buffer* buf, struct pgsd_handle* handle)
    {
#if PGSD_ACTIVATE_LOGGER
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("[INFO]: Rank: %i -> PGSD: pgsd_index_buffer_map\n", rank);
#endif
    if (buf == NULL || buf->mapped_data || buf->data || buf->reserved != 0 || buf->size != 0)
        {
        return PGSD_ERROR_INVALID_ARGUMENT;
        }

    // validate that the index block exists inside the file
    if (handle->header.index_location
            + sizeof(struct pgsd_index_entry) * handle->header.index_allocated_entries
        > (uint64_t)handle->file_size)
        {
        return PGSD_ERROR_FILE_CORRUPT;
        }

#if PGSD_USE_MMAP
    // map the index in read only mode
    size_t page_size = getpagesize();
    size_t index_size = sizeof(struct pgsd_index_entry) * handle->header.index_allocated_entries;
    size_t offset = (handle->header.index_location / page_size) * page_size;
    buf->mapped_data = mmap(NULL,
                            index_size + (handle->header.index_location - offset),
                            PROT_READ,
                            MAP_SHARED,
                            handle->fh,
                            offset);

    if (buf->mapped_data == MAP_FAILED)
        {
        return PGSD_ERROR_IO;
        }

    buf->data = (struct pgsd_index_entry*)(((char*)buf->mapped_data)
                                          + (handle->header.index_location - offset));

    buf->mapped_len = index_size + (handle->header.index_location - offset);
    buf->reserved = handle->header.index_allocated_entries;
#else
    // mmap not supported, read the data from the disk
    int retval = pgsd_index_buffer_allocate(buf, handle->header.index_allocated_entries);
    if (retval != PGSD_SUCCESS)
        {
        return retval;
        }
    MPI_File_seek(handle->fh, handle->header.index_location, MPI_SEEK_SET);
    MPI_File_read_at(handle->fh, handle->header.index_location, buf->data, sizeof(struct pgsd_index_entry)* handle->header.index_allocated_entries, MPI_BYTE, MPI_STATUS_IGNORE);

#endif

    // determine the number of index entries in the list
    // file is corrupt if first index entry is invalid
    if (buf->data[0].location != 0 && !pgsd_is_entry_valid(handle, 0))
        {
        return PGSD_ERROR_FILE_CORRUPT;
        }

    if (buf->data[0].location == 0)
        {
        buf->size = 0;
        }
    else
        {
        // determine the number of index entries (marked by location = 0)
        // binary search for the first index entry with location 0
        size_t L = 0;
        size_t R = buf->reserved;

        // progressively narrow the search window by halves
        do
            {
            size_t m = (L + R) / 2;
            
            // file is corrupt if any index entry is invalid or frame does not increase
            // monotonically
            if (buf->data[m].location != 0
                && (!pgsd_is_entry_valid(handle, m) || buf->data[m].frame < buf->data[L].frame))
                {
                return PGSD_ERROR_FILE_CORRUPT;
                }

            if (buf->data[m].location != 0)
                {
                L = m;
                }
            else
                {
                R = m;
                }
            } while ((R - L) > 1);

        // this finds R = the first index entry with location = 0
        buf->size = R;
        }

    return PGSD_SUCCESS;
    }

/** @internal
    @brief Free the memory allocated by the index buffer or unmap the mapped memory.

    @param buf Buffer to free.

    @returns PGSD_SUCCESS on success, PGSD_* error codes on error.

    DK : only executed on root
*/
inline static int pgsd_index_buffer_free(struct pgsd_index_buffer* buf)
    {
#if PGSD_ACTIVATE_LOGGER
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("[INFO]: Rank: %i -> PGSD: pgsd_index_buffer_free\n", rank);
#endif
    if (buf == NULL || buf->data == NULL)
        {
        return PGSD_ERROR_INVALID_ARGUMENT;
        }

#if PGSD_USE_MMAP
    if (buf->mapped_data)
        {
        int retval = munmap(buf->mapped_data, buf->mapped_len);

        if (retval != 0)
            {
            return PGSD_ERROR_IO;
            }
        }
    else
#endif
        {
        free(buf->data);
        }

    pgsd_util_zero_memory(buf, sizeof(struct pgsd_index_buffer));
    return PGSD_SUCCESS;
    }

/** @internal
    @brief Add a new index entry and provide a pointer to it.

    @param buf Buffer to add too.
    @param entry [out] Pointer to set to the new entry.

    Double the size of the reserved space as needed to hold the new entry. Does not accept mapped
    indices.

    @returns PGSD_SUCCESS on success, PGSD_* error codes on error.

    DK : only used on rank 0 since:
        N_global and M_global are entries in index (see write chunk)
*/
inline static int pgsd_index_buffer_add(struct pgsd_index_buffer* buf, struct pgsd_index_entry** entry)
    {
#if PGSD_ACTIVATE_LOGGER
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("[INFO]: Rank: %i -> PGSD: pgsd_index_buffer_add\n", rank);
#endif
    if (buf == NULL || buf->mapped_data || entry == NULL || buf->reserved == 0)
        {
        return PGSD_ERROR_INVALID_ARGUMENT;
        }

    if (buf->size == buf->reserved)
        {
        // grow the array
        size_t new_reserved = buf->reserved * 2;
        buf->data = realloc(buf->data, sizeof(struct pgsd_index_entry) * new_reserved);
        if (buf->data == NULL)
            {
            return PGSD_ERROR_MEMORY_ALLOCATION_FAILED;
            }

        // zero the new memory
        pgsd_util_zero_memory(buf->data + buf->reserved,
                             sizeof(struct pgsd_index_entry) * (new_reserved - buf->reserved));
        buf->reserved = new_reserved;
        }

    size_t insert_pos = buf->size;
    buf->size++;
    *entry = buf->data + insert_pos;

    return PGSD_SUCCESS;
    }

inline static int pgsd_cmp_index_entry(const struct pgsd_index_entry* a,
                                      const struct pgsd_index_entry* b)
    {
    int result = 0;

    if (a->frame < b->frame)
        {
        result = -1;
        }

    if (a->frame > b->frame)
        {
        result = 1;
        }

    if (a->frame == b->frame)
        {
        if (a->id < b->id)
            {
            result = -1;
            }

        if (a->id > b->id)
            {
            result = 1;
            }

        if (a->id == b->id)
            {
            result = 0;
            }
        }

    return result;
    }

/** @internal
    @brief Compute heap parent node.
    @param i Node index.
*/
inline static size_t pgsd_heap_parent(size_t i)
    {
    return (i - 1) / 2;
    }

/** @internal
    @brief Compute heap left child.
    @param i Node index.
*/
inline static size_t pgsd_heap_left_child(size_t i)
    {
    return 2 * i + 1;
    }

/** @internal
    @brief Swap the nodes *a* and *b* in the buffer
    @param buf Buffer.
    @param a First index to swap.
    @param b Second index to swap.
*/
inline static void pgsd_heap_swap(struct pgsd_index_buffer* buf, size_t a, size_t b)
    {
    struct pgsd_index_entry tmp = buf->data[a];
    buf->data[a] = buf->data[b];
    buf->data[b] = tmp;
    }

/** @internal
    @brief Shift heap node downward
    @param buf Buffer.
    @param start First index of the valid heap in *buf*.
    @param end Last index of the valid hep in *buf*.
*/
inline static void pgsd_heap_shift_down(struct pgsd_index_buffer* buf, size_t start, size_t end)
    {
    size_t root = start;

    while (pgsd_heap_left_child(root) <= end)
        {
        size_t child = pgsd_heap_left_child(root);
        size_t swap = root;

        if (pgsd_cmp_index_entry(buf->data + swap, buf->data + child) < 0)
            {
            swap = child;
            }
        if (child + 1 <= end && pgsd_cmp_index_entry(buf->data + swap, buf->data + child + 1) < 0)
            {
            swap = child + 1;
            }

        if (swap == root)
            {
            return;
            }

        pgsd_heap_swap(buf, root, swap);
        root = swap;
        }
    }

/** @internal
    @brief Convert unordered index buffer to a heap
    @param buf Buffer.
*/
inline static void pgsd_heapify(struct pgsd_index_buffer* buf)
    {
    ssize_t start = pgsd_heap_parent(buf->size - 1);

    while (start >= 0)
        {
        pgsd_heap_shift_down(buf, start, buf->size - 1);
        start--;
        }
    }

/** @internal
    @brief Sort the index buffer.

    @param buf Buffer to sort.

    Sorts an in-memory index buffer. Does not accept mapped indices.

    @returns PGSD_SUCCESS on success, PGSD_* error codes on error.
*/
inline static int pgsd_index_buffer_sort(struct pgsd_index_buffer* buf)
    {
#if PGSD_ACTIVATE_LOGGER
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("[INFO]: Rank: %i -> PGSD: pgsd_index_buffer_sort\n", rank);
#endif
    if (buf == NULL || buf->mapped_data || buf->reserved == 0)
        {
        return PGSD_ERROR_INVALID_ARGUMENT;
        }

    // arrays of size 0 or 1 are already sorted
    if (buf->size <= 1)
        {
        return PGSD_SUCCESS;
        }

    pgsd_heapify(buf);

    size_t end = buf->size - 1;
    while (end > 0)
        {
        pgsd_heap_swap(buf, end, 0);
        end = end - 1;
        pgsd_heap_shift_down(buf, 0, end);
        }

    return PGSD_SUCCESS;
    }

/** @internal
    @brief Utility function to expand the memory space for the index block in the file.

    @param handle Handle to the open pgsd file.
    @param size_required The new index must be able to hold at least this many elements.

    @returns PGSD_SUCCESS on success, PGSD_* error codes on error.

    DK : only used in flush with root loop 
*/
inline static int pgsd_expand_file_index(struct pgsd_handle* handle, size_t size_required)
    {
#if PGSD_ACTIVATE_LOGGER
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("[INFO]: Rank: %i -> PGSD: pgsd_expand_file_index\n", rank);
#endif    
    if (handle->open_flags == PGSD_OPEN_READONLY)
        {
        return PGSD_ERROR_FILE_MUST_BE_WRITABLE;
        }

    // multiply the index size each time it grows
    // this allows the index to grow rapidly to accommodate new frames
    const int multiplication_factor = 2;

    // save the old size and update the new size
    size_t size_old = handle->header.index_allocated_entries;
    size_t size_new = size_old * multiplication_factor;

    while (size_new <= size_required)
        {
        size_new *= multiplication_factor;
        }

    // Mac systems deadlock when writing from a mapped region into the tail end of that same region
    // unmap the index first and copy it over by chunks
    int retval = pgsd_index_buffer_free(&handle->file_index);

    if (retval != 0)
        {
        return retval;
        }

    // allocate the copy buffer
    uint64_t copy_buffer_size
        = PGSD_DEFAULT_INDEX_ENTRIES_TO_BUFFER * sizeof(struct pgsd_index_entry);
    if (copy_buffer_size > size_old * sizeof(struct pgsd_index_entry))
        {
        copy_buffer_size = size_old * sizeof(struct pgsd_index_entry);
        }
    char* buf = malloc(copy_buffer_size);
    if (buf == NULL)
        {
        return PGSD_ERROR_MEMORY_ALLOCATION_FAILED;
        }

    // write the current index to the end of the file
    long long int new_index_location;
    int64_t old_index_location = handle->header.index_location;
    
    MPI_File_get_size(handle->fh, &new_index_location);
    MPI_File_seek(handle->fh, 0, MPI_SEEK_END);

    size_t total_bytes_written = 0;
    size_t old_index_bytes = size_old * sizeof(struct pgsd_index_entry);

    while (total_bytes_written < old_index_bytes)
        {
        size_t bytes_to_copy = copy_buffer_size;
        if (old_index_bytes - total_bytes_written < copy_buffer_size)
            {
            bytes_to_copy = old_index_bytes - total_bytes_written;
            }

        MPI_File_read_at(handle->fh, old_index_location + total_bytes_written, buf, bytes_to_copy, MPI_BYTE, MPI_STATUS_IGNORE);

        if( is_root() ){
            MPI_File_write_at(handle->fh, new_index_location + total_bytes_written, buf, bytes_to_copy, MPI_BYTE, MPI_STATUS_IGNORE);
#if PGSD_ACTIVATE_LOGGER
            printf("[INFO]: Rank: %i -> PGSD: write at 1: loc %lli, size %li\n", rank, new_index_location + total_bytes_written, bytes_to_copy);
#endif
        }

        total_bytes_written += bytes_to_copy;
        }

    // fill the new index space with 0s
    pgsd_util_zero_memory(buf, copy_buffer_size);

    size_t new_index_bytes = size_new * sizeof(struct pgsd_index_entry);
    while (total_bytes_written < new_index_bytes)
        {
        size_t bytes_to_copy = copy_buffer_size;

        if (new_index_bytes - total_bytes_written < copy_buffer_size)
            {
            bytes_to_copy = new_index_bytes - total_bytes_written;
            }

        if( is_root() ){
            MPI_File_write_at(handle->fh, new_index_location + total_bytes_written, buf, bytes_to_copy, MPI_BYTE, MPI_STATUS_IGNORE);
#if PGSD_ACTIVATE_LOGGER
            printf("[INFO]: Rank: %i -> PGSD: write at 2: loc %lli, size %li\n", rank, new_index_location + total_bytes_written, bytes_to_copy);
#endif
        }

        total_bytes_written += bytes_to_copy;
        }

    // free the copy buffer
    free(buf);

    // update the header
    handle->header.index_location = new_index_location;
    handle->file_size = handle->header.index_location + total_bytes_written;
    handle->header.index_allocated_entries = size_new;

    // write the new header out
    if( is_root() ){
        MPI_File_write_at(handle->fh, 0, &(handle->header), sizeof(struct pgsd_header), MPI_BYTE, MPI_STATUS_IGNORE);
#if PGSD_ACTIVATE_LOGGER
        printf("[INFO]: Rank: %i -> PGSD: write at 3: loc 0, size %li\n", rank, sizeof(struct pgsd_header));
#endif
    }

    // remap the file index
    if ( is_root() )
        {
        retval = pgsd_index_buffer_map(&handle->file_index, handle);
        if (retval != 0)
            {
            return retval;
            }
        }

    return PGSD_SUCCESS;
    }

/** @internal
    @brief Flush the write buffer.

    pgsd_write_frame() writes small data chunks into the write buffer. It adds index entries for
    these chunks to pgsd_handle::buffer_index with locations offset from the start of the write
    buffer. pgsd_flush_write_buffer() writes the buffer to the end of the file, moves the index
    entries to pgsd_handle::frame_index and updates the location to reference the beginning of the
    file.

    @param handle Handle to flush the write buffer.
    @returns PGSD_SUCCESS on success or PGSD_* error codes on error

    DK : collectively. Use write buffer sizes as offset. 

*/
inline static int pgsd_flush_write_buffer(struct pgsd_handle* handle)
    {
    if (handle == NULL)
        {
        return PGSD_ERROR_INVALID_ARGUMENT;
        }

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
#if PGSD_ACTIVATE_LOGGER
    printf("[INFO]: Rank: %i -> PGSD: pgsd_flush_write_buffer\n", rank);
#endif

    size_t allbuffers[nprocs];
    MPI_Allgather(&handle->write_buffer.size, 1, my_MPI_SIZE_T, allbuffers, 1, my_MPI_SIZE_T, MPI_COMM_WORLD);

    if (handle->write_buffer.size == 0 && handle->buffer_index.size == 0)
        {
        // nothing to do
        return PGSD_SUCCESS;
        }

    if (is_root() && handle->write_buffer.size > 0 && handle->buffer_index.size == 0)
        {
        // error: bytes in buffer, but no index for them
#if PGSD_ACTIVATE_LOGGER
        printf("Rank %i: ERRORHANDLER: pgsd_flush_write_buffer -> Invalid argument!\n", rank);
#endif
        return PGSD_ERROR_INVALID_ARGUMENT;
        }
    bcast_file_size(handle);
    long long unsigned int offset = handle->file_size;

    // write the buffer to the end of the file
    // add buffers of smaller ranks
    int j;
    for( j = 0; j < rank; j++ ){
        offset += allbuffers[j];
    }

    MPI_File_write_at(handle->fh, offset, handle->write_buffer.data, handle->write_buffer.size, MPI_BYTE, MPI_STATUS_IGNORE);
#if PGSD_ACTIVATE_LOGGER
    printf("[INFO]: Rank: %i -> PGSD: write at 4: loc %lli, size %li\n", rank, offset, handle->write_buffer.size);
#endif

    // update the file_size in the handle
    MPI_Barrier(MPI_COMM_WORLD);
    size_t global_write_buffer_size;
    MPI_Allreduce(&handle->write_buffer.size, &global_write_buffer_size, 1, my_MPI_SIZE_T, MPI_SUM, MPI_COMM_WORLD);
    
    if ( is_root() )
        {
        handle->file_size += global_write_buffer_size;
        }
    handle->write_buffer.size = 0;

    // Broadcast the new file size
    bcast_file_size(handle);

    // reset write_buffer for new data
    // Move buffer_index entries to frame_index.
    if ( is_root() )
        {
        size_t i;
        for (i = 0; i < handle->buffer_index.size; i++)
            {
            struct pgsd_index_entry* new_index;
            int retval = pgsd_index_buffer_add(&handle->frame_index, &new_index);
            if (retval != PGSD_SUCCESS)
                {
#if PGSD_ACTIVATE_LOGGER
                printf("Rank %i: ERRORHANDLER: pgsd_flush_write_buffer -> pgsd_index_buffer_add!\n", rank);
#endif
                return retval;
                }

            *new_index = handle->buffer_index.data[i];
            new_index->location += offset;
            }
        }

    // clear the buffer index for new entries
    handle->buffer_index.size = 0;

    return PGSD_SUCCESS;
    }

/** @internal
    @brief Flush the name buffer.

    pgsd_write_frame() adds new names to the frame_names buffer. pgsd_flush_name_buffer() flushes
    this buffer at the end of a frame write and commits the new names to the file. If necessary,
    the namelist is written to a new location in the file.

    @param handle Handle to flush the write buffer.
    @returns PGSD_SUCCESS on success or PGSD_* error codes on error


    DK :
*/
inline static int pgsd_flush_name_buffer(struct pgsd_handle* handle)
    {
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
#if PGSD_ACTIVATE_LOGGER
    printf("[INFO]: Rank: %i -> PGSD: pgsd_flush_name_buffer\n", rank);
#endif
    int retval = 0;
    if (handle == NULL)
        {
        return PGSD_ERROR_INVALID_ARGUMENT;
        }

    size_t indicator_n_names  = handle->frame_names.n_names;
    MPI_Allreduce(MPI_IN_PLACE, &indicator_n_names, 1, my_MPI_SIZE_T, MPI_SUM, MPI_COMM_WORLD);

    if ( indicator_n_names == 0)
        {
        // nothing to do
        return PGSD_SUCCESS;
        }

    if( is_root() )
        { 
        if (handle->frame_names.data.size == 0)
            {
            // error: bytes in buffer, but no names for them
#if PGSD_ACTIVATE_LOGGER
            printf("Rank %i: ERRORHANDLER: pgsd_flush_name_buffer -> Invalid argument 0!\n", rank);
#endif
            return PGSD_ERROR_INVALID_ARGUMENT;
            }
        }
    size_t old_reserved = handle->file_names.data.reserved;
    size_t old_size = handle->file_names.data.size;

    if ( is_root() )
        {
        // add the new names to the file name list and zero the frame list
        retval = pgsd_byte_buffer_append(&handle->file_names.data,
                                            handle->frame_names.data.data,
                                            handle->frame_names.data.size);
        if (retval != PGSD_SUCCESS)
            {
#if PGSD_ACTIVATE_LOGGER
            printf("Rank %i: ERRORHANDLER: pgsd_flush_name_buffer -> Invalid argument 1!\n", rank);
#endif
            return retval;
            }
        handle->file_names.n_names += handle->frame_names.n_names;
        }

    handle->frame_names.n_names = 0;
    handle->frame_names.data.size = 0;
    bcast_number_of_names_file(handle);

    pgsd_util_zero_memory(handle->frame_names.data.data, handle->frame_names.data.reserved);

    if ( is_root() )
        {
        // reserved space must be a multiple of the PGSD name size
        if (handle->file_names.data.reserved % PGSD_NAME_SIZE != 0)
            {
#if PGSD_ACTIVATE_LOGGER
            printf("Rank %i: ERRORHANDLER: pgsd_flush_name_buffer -> Invalid argument 2!\n", rank);
#endif
            return PGSD_ERROR_INVALID_ARGUMENT;
            }

        if (handle->file_names.data.reserved > old_reserved)
            {
            // write the new name list to the end of the file
            long long unsigned int offset = handle->file_size;

            if ( is_root() ){
                MPI_File_write_at(handle->fh, offset, handle->file_names.data.data, handle->file_names.data.reserved, MPI_BYTE, MPI_STATUS_IGNORE);
#if PGSD_ACTIVATE_LOGGER
                printf("[INFO]: Rank: %i -> PGSD: write at 5: loc %lli, size %li\n", rank, offset, handle->file_names.data.reserved);
#endif
            }

            handle->file_size += handle->file_names.data.reserved;
            handle->header.namelist_location = offset;
            handle->header.namelist_allocated_entries
                = handle->file_names.data.reserved / PGSD_NAME_SIZE;

            // write the new header out
            if( is_root() ){
                MPI_File_write_at(handle->fh, 0, &(handle->header), sizeof(struct pgsd_header), MPI_BYTE, MPI_STATUS_IGNORE);
            }
            }
        else
            {
            // write the new name list to the old index location
            long long unsigned int offset = handle->header.namelist_location;
            if( is_root() )
                {
                MPI_File_write_at(handle->fh, offset + old_size, handle->file_names.data.data + old_size, handle->file_names.data.reserved - old_size, MPI_BYTE, MPI_STATUS_IGNORE);
#if PGSD_ACTIVATE_LOGGER
                printf("[INFO]: Rank: %i -> PGSD: write at 6: loc %lli, size %li\n", rank, offset+old_size, handle->file_names.data.reserved - old_size);
#endif
                }
            }
        }

    bcast_file_size(handle);
    bcast_namelist_location(handle);
    bcast_namelist_allocated_entries(handle);


    return PGSD_SUCCESS;
    }

/** @internal
    @brief utility function to append a name to the namelist

    @param id [out] ID of the new name
    @param handle handle to the open pgsd file
    @param name string name

    Append a name to the names in the current frame. pgsd_end_frame() will add this list to the
    file names.

    @return
      - PGSD_SUCCESS (0) on success. Negative value on failure:
      - PGSD_ERROR_IO: IO error (check errno).
      - PGSD_ERROR_MEMORY_ALLOCATION_FAILED: Unable to allocate memory.
      - PGSD_ERROR_FILE_MUST_BE_WRITABLE: File must not be read only.

    DK : Append name is only used within a root loop
         names should only be appended once
*/
inline static int pgsd_append_name(uint16_t* id, struct pgsd_handle* handle, const char* name)
    {
#if PGSD_ACTIVATE_LOGGER
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("[INFO]: Rank: %i -> PGSD: pgsd_append_name\n", rank);
#endif
    if (handle->open_flags == PGSD_OPEN_READONLY)
        {
#if PGSD_ACTIVATE_LOGGER
                printf("Rank %i: ERRORHANDLER: pgsd_append_name -> File must be writeable!\n", rank);
#endif
        return PGSD_ERROR_FILE_MUST_BE_WRITABLE;
        }

    if (handle->file_names.n_names + handle->frame_names.n_names == UINT16_MAX)
        {
        // no more names may be added
#if PGSD_ACTIVATE_LOGGER
                printf("Rank %i: ERRORHANDLER: pgsd_append_name -> Namelist full!\n", rank);
#endif
        return PGSD_ERROR_NAMELIST_FULL;
        }

    // Provide the ID of the new name
    *id = (uint16_t)(handle->file_names.n_names + handle->frame_names.n_names);

    if (handle->header.pgsd_version < pgsd_make_version(2, 0))
        {
        // v1 files always allocate PGSD_NAME_SIZE bytes for each name and put a NULL terminator
        // at address 63
        char name_v1[PGSD_NAME_SIZE];
        strncpy(name_v1, name, PGSD_NAME_SIZE - 1);
        name_v1[PGSD_NAME_SIZE - 1] = 0;
        pgsd_byte_buffer_append(&handle->frame_names.data, name_v1, PGSD_NAME_SIZE);
        handle->frame_names.n_names++;

        // update the name/id mapping with the truncated name
        int retval = pgsd_name_id_map_insert(&handle->name_map, name_v1, *id);
        if (retval != PGSD_SUCCESS)
            {
#if PGSD_ACTIVATE_LOGGER
            printf("Rank %i: ERRORHANDLER: pgsd_append_name -> pgsd_name_id_map_insert 1!\n", rank);
#endif
            return retval;
            }
        }
    else
        {
        pgsd_byte_buffer_append(&handle->frame_names.data, name, strlen(name) + 1);
        handle->frame_names.n_names++;

        // update the name/id mapping
        int retval = pgsd_name_id_map_insert(&handle->name_map, name, *id);
        if (retval != PGSD_SUCCESS)
            {
#if PGSD_ACTIVATE_LOGGER
            printf("Rank %i: ERRORHANDLER: pgsd_append_name -> pgsd_name_id_map_insert 2!\n", rank);
#endif
            return retval;
            }
        }

    return PGSD_SUCCESS;
    }

/** @internal
    @brief Truncate the file and write a new pgsd header.

    @param fd file descriptor to initialize
    @param application Generating application name (truncated to 63 chars)
    @param schema Schema name for data to be written in this PGSD file (truncated to 63 chars)
    @param schema_version Version of the scheme data to be written (make with pgsd_make_version())
*/
inline static int
pgsd_initialize_file(MPI_File fh, const char* application, const char* schema, uint32_t schema_version)
    {
#if PGSD_ACTIVATE_LOGGER
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    printf("[INFO]: Rank: %i -> PGSD: pgsd_initialize_file\n", rank);
#endif

    // truncate the file and verify success
    MPI_Offset file_size = 0;
    int retval = MPI_File_set_size(fh, file_size);
    MPI_File_seek(fh, 0, MPI_SEEK_SET);

    if (retval != MPI_SUCCESS)
        {
        return PGSD_ERROR_IO;
        }
    if( is_root() )
    {
        // populate header fields
        struct pgsd_header header;
        pgsd_util_zero_memory(&header, sizeof(header));

        header.magic = PGSD_MAGIC_ID;
        header.pgsd_version = pgsd_make_version(PGSD_CURRENT_FILE_VERSION, 0);
        strncpy(header.application, application, sizeof(header.application) - 1);
        header.application[sizeof(header.application) - 1] = 0;
        strncpy(header.schema, schema, sizeof(header.schema) - 1);
        header.schema[sizeof(header.schema) - 1] = 0;
        header.schema_version = schema_version;
        header.index_location = sizeof(header);
        header.index_allocated_entries = PGSD_INITIAL_INDEX_SIZE;
        header.namelist_location
            = header.index_location + sizeof(struct pgsd_index_entry) * header.index_allocated_entries;
        header.namelist_allocated_entries = PGSD_INITIAL_NAME_BUFFER_SIZE / PGSD_NAME_SIZE;
        
        // Write header 
        pgsd_util_zero_memory(header.reserved, sizeof(header.reserved));

        MPI_File_write(fh, &header, sizeof(header), MPI_BYTE, MPI_STATUS_IGNORE);

        // allocate and zero default index memory
        struct pgsd_index_entry index[PGSD_INITIAL_INDEX_SIZE];
        pgsd_util_zero_memory(index, sizeof(index));
            
        MPI_File_write(fh, &index, sizeof(index), MPI_BYTE, MPI_STATUS_IGNORE);
            

        // allocate and zero the namelist memory
        char names[PGSD_INITIAL_NAME_BUFFER_SIZE];
        pgsd_util_zero_memory(names, sizeof(char) * PGSD_INITIAL_NAME_BUFFER_SIZE);

        MPI_File_write(fh, &names, sizeof(names), MPI_BYTE, MPI_STATUS_IGNORE);

    }
    MPI_Barrier(MPI_COMM_WORLD);
    return PGSD_SUCCESS;
    }

/** @internal
    @brief Read in the file index and initialize the handle.

    @param handle Handle to read the header

    @pre handle->fd is an open file.
    @pre handle->open_flags is set.
*/
inline static int 
pgsd_initialize_handle(struct pgsd_handle* handle)
    {
#if PGSD_ACTIVATE_LOGGER
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("[INFO]: Rank: %i -> PGSD: pgsd_initialize_handle\n", rank);
#endif
    // check if the file was created
    if (handle->fh == NULL)
        {
        return PGSD_ERROR_IO;
        }

    // read the header at all ranks
    MPI_File_seek(handle->fh, 0, MPI_SEEK_SET);
    MPI_File_read(handle->fh, &(handle->header), sizeof(struct pgsd_header), MPI_BYTE, MPI_STATUS_IGNORE);

    // validate the header
    if (handle->header.magic != PGSD_MAGIC_ID)
        {
        return PGSD_ERROR_NOT_A_PGSD_FILE;
        }
    if (handle->header.pgsd_version < pgsd_make_version(1, 0)
        && handle->header.pgsd_version != pgsd_make_version(0, 3))
        {
        return PGSD_ERROR_INVALID_PGSD_FILE_VERSION;
        }
    if (handle->header.pgsd_version >= pgsd_make_version(3, 0))
        {
        return PGSD_ERROR_INVALID_PGSD_FILE_VERSION;
        }

    // determine the file size
    MPI_File_get_size(handle->fh, &(handle->file_size));
    MPI_File_seek(handle->fh, 0, MPI_SEEK_END);

    int retval = 0;
    // validate that the namelist block exists inside the file
    if (handle->header.namelist_location
            + (PGSD_NAME_SIZE * handle->header.namelist_allocated_entries)
        > (uint64_t)handle->file_size)
        {
        return PGSD_ERROR_FILE_CORRUPT;
        }

    if ( is_root() )
        {
        // allocate the hash map
        retval = pgsd_name_id_map_allocate(&handle->name_map, PGSD_NAME_MAP_SIZE);
    
        if (retval != PGSD_SUCCESS)
            {

#if PGSD_ACTIVATE_LOGGER
            printf("Rank %i: ERRORHANDLER: pgsd_initialize_handle -> pgsd_name_id_map_allocate!\n", rank);
#endif

            return retval;
            }

        // read the namelist block
        size_t namelist_n_bytes = PGSD_NAME_SIZE * handle->header.namelist_allocated_entries;
        retval = pgsd_byte_buffer_allocate(&handle->file_names.data, namelist_n_bytes);
        if (retval != PGSD_SUCCESS)
            {

#if PGSD_ACTIVATE_LOGGER
            printf("Rank %i: ERRORHANDLER: pgsd_initialize_handle -> pgsd_byte_buffer_allocate!\n", rank);
#endif
            
            return retval;
            }

        MPI_File_read_at(handle->fh, handle->header.namelist_location, handle->file_names.data.data, namelist_n_bytes, MPI_BYTE, MPI_STATUS_IGNORE);

        // The name buffer must end in a NULL terminator or else the file is corrupt
        if (handle->file_names.data.data[handle->file_names.data.reserved - 1] != 0)
            {

#if PGSD_ACTIVATE_LOGGER
            printf("Rank %i: ERRORHANDLER: pgsd_initialize_handle -> Corrupt reserved data!\n", rank);
#endif
            return PGSD_ERROR_FILE_CORRUPT;
            }

        // Add the names to the hash map. Also determine the number of used bytes in the namelist.
        size_t name_start = 0;
        handle->file_names.n_names = 0;
        while (name_start < handle->file_names.data.reserved)
            {
            char* name = handle->file_names.data.data + name_start;

            // an empty name notes the end of the list
            if (name[0] == 0)
                {
                break;
                }

            retval
                = pgsd_name_id_map_insert(&handle->name_map, name, (uint16_t)handle->file_names.n_names);
            if (retval != PGSD_SUCCESS)
                {
                return retval;
#if PGSD_ACTIVATE_LOGGER
                printf("Rank %i: ERRORHANDLER: pgsd_initialize_handle -> pgsd_name_id_map_insert!\n", rank);
#endif
                }
            handle->file_names.n_names++;

            if (handle->header.pgsd_version < pgsd_make_version(2, 0))
                {
                // pgsd v1 stores names in fixed 64 byte segments
                name_start += PGSD_NAME_SIZE;
                }
            else
                {
                size_t len = strnlen(name, handle->file_names.data.reserved - name_start);
                name_start += len + 1;
                }
            }
        handle->file_names.data.size = name_start;
        }

    if ( !file_name_size_is_same(handle->file_names.data.size))
    {
        MPI_Bcast(&handle->file_names.data.size, 1, my_MPI_SIZE_T, 0, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    // read in the file index
    if ( is_root() )
        {
        retval = pgsd_index_buffer_map(&handle->file_index, handle);
        if (retval != PGSD_SUCCESS)
            {

#if PGSD_ACTIVATE_LOGGER
            printf("Rank %i: ERRORHANDLER: pgsd_initialize_handle -> pgsd_index_buffer_map!\n", rank);
#endif

            return retval;
            }

        // determine the current frame counter
        if (handle->file_index.size == 0)
            {
            handle->cur_frame = 0;
            }
        else
            {
            handle->cur_frame = handle->file_index.data[handle->file_index.size - 1].frame + 1;
            }
        }
    MPI_Bcast(&handle->cur_frame, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    // if this is a write mode, allocate the initial frame index and the name buffer
    // allocate frame index, buffer index on root
    if ( is_root() )
        {
        if (handle->open_flags != PGSD_OPEN_READONLY)
            {
            retval = pgsd_index_buffer_allocate(&handle->frame_index, PGSD_INITIAL_FRAME_INDEX_SIZE);
            if (retval != PGSD_SUCCESS)
                {

#if PGSD_ACTIVATE_LOGGER
            printf("Rank %i: ERRORHANDLER: pgsd_initialize_handle -> pgsd_index_buffer_allocate frame!\n", rank);
#endif
                return retval;
                }

            retval = pgsd_index_buffer_allocate(&handle->buffer_index, PGSD_INITIAL_FRAME_INDEX_SIZE);
            if (retval != PGSD_SUCCESS)
                {

#if PGSD_ACTIVATE_LOGGER
            printf("Rank %i: ERRORHANDLER: pgsd_initialize_handle -> pgsd_index_buffer_allocate buffer!\n", rank);
#endif
                return retval;
                }
            }
        }

    if (handle->open_flags != PGSD_OPEN_READONLY)
        {
        retval = pgsd_byte_buffer_allocate(&handle->write_buffer, PGSD_INITIAL_WRITE_BUFFER_SIZE);
        if (retval != PGSD_SUCCESS)
            {

#if PGSD_ACTIVATE_LOGGER
            printf("Rank %i: ERRORHANDLER: pgsd_initialize_handle -> pgsd_byte_buffer_allocate write!\n", rank);
#endif
            return retval;
            }
        handle->frame_names.n_names = 0;
    
        if ( is_root() )
            {  
            retval = pgsd_byte_buffer_allocate(&handle->frame_names.data, PGSD_NAME_SIZE);
            if (retval != PGSD_SUCCESS)
                {

#if PGSD_ACTIVATE_LOGGER
                printf("Rank %i: ERRORHANDLER: pgsd_initialize_handle -> pgsd_byte_buffer_allocate names!\n", rank);
#endif

                return retval;
                }
            }
        }

    handle->pending_index_entries = 0;
    handle->maximum_write_buffer_size = PGSD_DEFAULT_MAXIMUM_WRITE_BUFFER_SIZE;
    handle->index_entries_to_buffer = PGSD_DEFAULT_INDEX_ENTRIES_TO_BUFFER;
    bcast_number_of_names_file(handle);

    return PGSD_SUCCESS;
    }

uint32_t pgsd_make_version(unsigned int major, unsigned int minor)
    {
    return major << (sizeof(uint32_t) * 4) | minor;
    }

int pgsd_create_and_open(struct pgsd_handle* handle,
                        const char* fname,
                        const char* application,
                        const char* schema,
                        uint32_t schema_version,
                        const enum pgsd_open_flag flags,
                        int exclusive_create)
    {
#if PGSD_ACTIVATE_LOGGER
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("[INFO]: Rank: %i -> PGSD: pgsd_create_and_open\n", rank);
#endif
    // zero the handle
    pgsd_util_zero_memory(handle, sizeof(struct pgsd_handle));

    int extra_flags = 0;

    // set the open flags in the handle
    if (flags == PGSD_OPEN_READWRITE)
        {
        handle->open_flags = PGSD_OPEN_READWRITE;
        }
    else if (flags == PGSD_OPEN_READONLY)
        {
        return PGSD_ERROR_FILE_MUST_BE_WRITABLE;
        }
    else if (flags == PGSD_OPEN_APPEND)
        {
        handle->open_flags = PGSD_OPEN_APPEND;
        }

    // set the exclusive create bit
    if (exclusive_create)
        {
        extra_flags |= MPI_MODE_EXCL;
        }

    MPI_File_open(MPI_COMM_WORLD, fname, MPI_MODE_RDWR | MPI_MODE_CREATE | extra_flags, MPI_INFO_NULL, &(handle->fh));

    int retval = pgsd_initialize_file(handle->fh, application, schema, schema_version);
    bcast_retval(&retval);

    if (retval != 0)
        {
        if (retval != -1)
            {
            MPI_File_close(&(handle->fh));
            }
        return retval;
        }

    retval = pgsd_initialize_handle(handle);
    bcast_retval(&retval);
    if (retval != 0)
        {
        if (handle->fh != NULL)
            {
            MPI_File_close(&(handle->fh));
            }
        }

    return retval;
    }

int pgsd_open(struct pgsd_handle* handle, const char* fname, const enum pgsd_open_flag flags)
    {
#if PGSD_ACTIVATE_LOGGER
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("[INFO]: Rank: %i -> PGSD: pgsd_open\n", rank);
#endif
    // zero the handle
    pgsd_util_zero_memory(handle, sizeof(struct pgsd_handle));

    // open the file
    if (flags == PGSD_OPEN_READWRITE)
        {
        MPI_File_open(MPI_COMM_WORLD ,fname, MPI_MODE_RDWR, MPI_INFO_NULL, &(handle->fh));
        handle->open_flags = PGSD_OPEN_READWRITE;
        }
    else if (flags == PGSD_OPEN_READONLY)
        {
        MPI_File_open(MPI_COMM_WORLD ,fname, MPI_MODE_RDONLY, MPI_INFO_NULL, &(handle->fh));
        handle->open_flags = PGSD_OPEN_READONLY;
        }
    else if (flags == PGSD_OPEN_APPEND)
        {
        MPI_File_open(MPI_COMM_WORLD ,fname, MPI_MODE_RDWR, MPI_INFO_NULL, &(handle->fh));
        handle->open_flags = PGSD_OPEN_APPEND;
        }

    int retval = pgsd_initialize_handle(handle);
    if (retval != 0)
        {
        if (handle->fh != NULL)
            {
            MPI_File_close(&(handle->fh));
            }
        }

    return retval;
    }

int pgsd_close(struct pgsd_handle* handle)
    {
#if PGSD_ACTIVATE_LOGGER
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("[INFO]: Rank: %i -> PGSD: pgsd_close\n", rank);
#endif
    if (handle == NULL)
        {
        return PGSD_ERROR_INVALID_ARGUMENT;
        }

    int retval = 0;

    if (handle->open_flags != PGSD_OPEN_READONLY)
        {
        retval = pgsd_flush(handle);
        if (retval != PGSD_SUCCESS)
            {
            return retval;
            }
        }

    MPI_Barrier(MPI_COMM_WORLD);

    if ( is_root() )
        {
        retval = pgsd_index_buffer_free(&handle->file_index);
        if (retval != PGSD_SUCCESS)
            {
            return retval;
            }
        
        if (handle->frame_index.reserved > 0)
            {
            retval = pgsd_index_buffer_free(&handle->frame_index);
            if (retval != PGSD_SUCCESS)
                {
                return retval;
                }
            }

        if (handle->buffer_index.reserved > 0)
            {
            retval = pgsd_index_buffer_free(&handle->buffer_index);
            if (retval != PGSD_SUCCESS)
                {
                return retval;
                }
            }
        }

    if (handle->write_buffer.reserved > 0)
        {
        retval = pgsd_byte_buffer_free(&handle->write_buffer);
        if (retval != PGSD_SUCCESS)
            {
            return retval;
            }
        }

    if ( is_root() )
        {
        retval = pgsd_name_id_map_free(&handle->name_map);
        if (retval != PGSD_SUCCESS)
            {
            return retval;
            }

        if (handle->frame_names.data.reserved > 0)
            {
            handle->frame_names.n_names = 0;
            retval = pgsd_byte_buffer_free(&handle->frame_names.data);
            if (retval != PGSD_SUCCESS)
                {
                return retval;
                }
            }

        if (handle->file_names.data.reserved > 0)
            {
            handle->file_names.n_names = 0;
            retval = pgsd_byte_buffer_free(&handle->file_names.data);
            if (retval != PGSD_SUCCESS)
                {
                return retval;
                }
            }
        }

    // close the file
    MPI_Barrier(MPI_COMM_WORLD);
    retval = MPI_File_close(&(handle->fh));

    if (retval != 0)
        {
        return PGSD_ERROR_IO;
        }

    return PGSD_SUCCESS;
    }

int pgsd_end_frame(struct pgsd_handle* handle)
    {
#if PGSD_ACTIVATE_LOGGER
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("[INFO]: Rank: %i -> PGSD: pgsd_end_frame\n", rank);
#endif
    if (handle == NULL)
        {
        return PGSD_ERROR_INVALID_ARGUMENT;
        }
    if (handle->open_flags == PGSD_OPEN_READONLY)
        {
        return PGSD_ERROR_FILE_MUST_BE_WRITABLE;
        }

    MPI_Barrier(MPI_COMM_WORLD);

    // increment the frame counter
    handle->cur_frame++;
    handle->pending_index_entries = 0;

    if ( !frame_is_same(handle->cur_frame) )
        fprintf(stderr, "Frame numbers are not the same!\n");
    
    int flush_indicator = 0;
    if (handle->frame_index.size > 0 || handle->buffer_index.size > handle->index_entries_to_buffer)
    {
        flush_indicator = 1;
    }
    MPI_Allreduce(MPI_IN_PLACE, &flush_indicator, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if ( flush_indicator != 0)
        {
        return pgsd_flush(handle);
        }

    return PGSD_SUCCESS;
    }

int pgsd_flush(struct pgsd_handle* handle)
    {
#if PGSD_ACTIVATE_LOGGER
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("[INFO]: Rank: %i -> PGSD: pgsd_flush\n", rank);
#endif
    MPI_Barrier(MPI_COMM_WORLD);

    if (handle == NULL)
        {
        return PGSD_ERROR_INVALID_ARGUMENT;
        }
    if (handle->open_flags == PGSD_OPEN_READONLY)
        {
        return PGSD_ERROR_FILE_MUST_BE_WRITABLE;
        }


    // flush the namelist buffer
    // actual flushing is only done on root, but communication makes is necessary 
    // that all ranks enter the function
    int retval = 0;
    retval = pgsd_flush_name_buffer(handle);
    if (retval != PGSD_SUCCESS)
        {
#if PGSD_ACTIVATE_LOGGER
        printf("Rank %i: ERRORHANDLER: pgsd_flush -> pgsd_flush_name_buffer!\n", rank);
#endif
        return retval;
        }

    MPI_Barrier(MPI_COMM_WORLD);
    // flush the write buffer
    retval = pgsd_flush_write_buffer(handle);
    if (retval != PGSD_SUCCESS)
        {
        return retval;
        }

    // Wait for all since index buffer is sorted in here
    MPI_Barrier(MPI_COMM_WORLD);

    // Write the frame index to the file, excluding the index entries that are part of the current
    // frame.
    if ( is_root() )
        {
        if (handle->pending_index_entries > handle->frame_index.size)
            {
#if PGSD_ACTIVATE_LOGGER
            printf("Rank %i: ERRORHANDLER: pgsd_flush -> Invalid argument!\n", rank);
#endif
            return PGSD_ERROR_INVALID_ARGUMENT;
            }
        uint64_t index_entries_to_write = handle->frame_index.size - handle->pending_index_entries;

        if (index_entries_to_write > 0)
            {
            // ensure there is enough space in the index
            if ((handle->file_index.size + index_entries_to_write) > handle->file_index.reserved)
                {
                pgsd_expand_file_index(handle, handle->file_index.size + index_entries_to_write);
                }

            // sort the index before writing
            retval = pgsd_index_buffer_sort(&handle->frame_index);
            if (retval != 0)
                {
#if PGSD_ACTIVATE_LOGGER
                printf("Rank %i: ERRORHANDLER: pgsd_flush -> pgsd_index_buffer_sort!\n", rank);
#endif
                return retval;
                }

            // write the frame index entries to the file
            long long unsigned int write_pos = handle->header.index_location
                                + sizeof(struct pgsd_index_entry) * handle->file_index.size;

            if ( is_root() ){
                MPI_File_write_at(handle->fh, write_pos, handle->frame_index.data, sizeof(struct pgsd_index_entry) * handle->frame_index.size, MPI_BYTE, MPI_STATUS_IGNORE);
#if PGSD_ACTIVATE_LOGGER
                printf("[INFO]: Rank: %i -> PGSD: write at 7: loc %lli, size %li\n", rank, write_pos, sizeof(struct pgsd_index_entry) * handle->frame_index.size);
#endif
            }

#if !PGSD_USE_MMAP
            // add the entries to the file index
            memcpy(handle->file_index.data + handle->file_index.size,
                   handle->frame_index.data,
        
                   sizeof(struct pgsd_index_entry) * handle->frame_index.size);
#endif

            // update size of file index
            handle->file_index.size += index_entries_to_write;

            // Clear the frame index, keeping those in the current unfinished frame.
            if (handle->pending_index_entries > 0)
                {
                for (uint64_t i = 0; i < handle->pending_index_entries; i++)
                    {
                    handle->frame_index.data[i]
                        = handle->frame_index
                              .data[handle->frame_index.size - handle->pending_index_entries];
                    }
                }

            handle->frame_index.size = handle->pending_index_entries;

            }
        }

    bcast_file_size(handle);
    bcast_index_location(handle);
    bcast_index_allocated_entries(handle);
    bcast_frame_index_size(handle);

    return PGSD_SUCCESS;
    }

int pgsd_write_chunk(struct pgsd_handle* handle,
                    const char* name,
                    enum pgsd_type type,
                    uint64_t N,
                    uint32_t M,
                    uint64_t N_global,
                    uint32_t M_global,
                    uint64_t offset,
                    uint64_t global_size,
                    bool all,
                    uint8_t flags,
                    const void* data)
    {
#if PGSD_ACTIVATE_LOGGER
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("[INFO]: Rank: %i -> PGSD: pgsd_write_chunk\n", rank);
#endif
    // validate input
    if (N > 0 && data == NULL)
        {
        return PGSD_ERROR_INVALID_ARGUMENT;
        }
    if (M == 0)
        {
        return PGSD_ERROR_INVALID_ARGUMENT;
        }
    if (handle->open_flags == PGSD_OPEN_READONLY)
        {
        return PGSD_ERROR_FILE_MUST_BE_WRITABLE;
        }
    if (flags != 0)
        {
        return PGSD_ERROR_INVALID_ARGUMENT;
        }
    
    uint16_t id;
    struct pgsd_index_entry entry;
    pgsd_util_zero_memory(&entry, sizeof(struct pgsd_index_entry));
    
    if ( is_root() )
        {
        id = pgsd_name_id_map_find(&handle->name_map, name);
        if (id == UINT16_MAX)
            {
            // not found, append to the index
            int retval = pgsd_append_name(&id, handle, name);
            if (retval != PGSD_SUCCESS)
                {
#if PGSD_ACTIVATE_LOGGER
                printf("Rank %i: ERRORHANDLER: pgsd_write_chunk -> pgsd_append_name!\n", rank);
#endif
                return retval;
                }
            if (id == UINT16_MAX)
                {
                // this should never happen
#if PGSD_ACTIVATE_LOGGER
                printf("Rank %i: ERRORHANDLER: pgsd_write_chunk -> Namelist full!\n", rank);
#endif
                return PGSD_ERROR_NAMELIST_FULL;
                }
            }

        // populate fields in the entry's data
        entry.frame = handle->cur_frame;
        entry.id = id;
        entry.type = (uint8_t)type;
        entry.N = N_global;
        entry.M = M_global;
        }
    
    bcast_number_of_names_frame(handle);
    
    size_t size = N * M * pgsd_sizeof_type(type);

    global_size *= pgsd_sizeof_type(type);
    offset *= pgsd_sizeof_type(type);
    if ( global_size == 0 && offset == 0 ){
        global_size = size;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Must not end up in different parts of the if clause due to barriers
    size_t maxsize;
    MPI_Allreduce(&size, &maxsize, 1, my_MPI_SIZE_T, MPI_MAX, MPI_COMM_WORLD);

    // decide whether to write this chunk to the buffer or straight to disk
    if (maxsize < handle->maximum_write_buffer_size && all == false)
        {
        // flush the buffer if this entry won't fit
        // collectively, inside the function there is a offset defined 
        // dependend on other rank's write buffers
        if (size > (handle->maximum_write_buffer_size - handle->write_buffer.size))
            {
            pgsd_flush_write_buffer(handle);
            }
        int retval = 0;
        // the location of write buffer of rank 0
        if ( is_root() )
            {
            entry.location = handle->write_buffer.size;

            // add an entry to the buffer index
            struct pgsd_index_entry* index_entry;

            retval = pgsd_index_buffer_add(&handle->buffer_index, &index_entry);
            if (retval != PGSD_SUCCESS)
                {
#if PGSD_ACTIVATE_LOGGER
                printf("Rank %i: ERRORHANDLER: pgsd_write_chunk -> pgsd_index_buffer_add!\n", rank);
#endif
                return retval;
                }
            *index_entry = entry;
            }


        // add the data to the write buffer
        if (size > 0)
            {
            retval = pgsd_byte_buffer_append(&handle->write_buffer, data, size);
            if (retval != PGSD_SUCCESS)
                {
#if PGSD_ACTIVATE_LOGGER
                printf("Rank %i: ERRORHANDLER: pgsd_write_chunk -> pgsd_byte_buffer_append!\n", rank);
#endif
                return retval;
                }
            }
        }
    else
        {
        if ( is_root() )
            {
            // add an entry to the frame index
            struct pgsd_index_entry* index_entry;

            int retval = pgsd_index_buffer_add(&handle->frame_index, &index_entry);
            if (retval != PGSD_SUCCESS)
                {
                return retval;
                }
            *index_entry = entry;
            index_entry->location = handle->file_size;
            }
        
        if ( !file_size_is_same(handle->file_size))
            {
            bcast_file_size(handle);
            }

        // find the location at the end of the file for the chunk
        long long int location_to_write = handle->file_size + offset;

        // write the data
        size_t bytes_written = size;
        if( all == true || is_root() ){
            MPI_File_write_at(handle->fh, location_to_write, data, size, MPI_BYTE, MPI_STATUS_IGNORE);
#if PGSD_ACTIVATE_LOGGER
            printf("[INFO]: Rank: %i -> PGSD: write at sim data: loc %lli, size %li\n", rank, location_to_write, size);
#endif
        }
        if (bytes_written == -1 || bytes_written != size)
            {
            return PGSD_ERROR_IO;
            }

        // update the file_size in the handle
        size_t global_bytes_written;
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(&bytes_written, &global_bytes_written, 1, my_MPI_SIZE_T, MPI_SUM, MPI_COMM_WORLD);
        
        if ( is_root() )
            {
            handle->file_size += global_bytes_written;
            }

        bcast_file_size(handle);
        }


    if ( is_root() )
        {
        handle->pending_index_entries++;
        }
    MPI_Barrier(MPI_COMM_WORLD);
    return PGSD_SUCCESS;
    }

uint64_t pgsd_get_nframes(struct pgsd_handle* handle)
    {
#if PGSD_ACTIVATE_LOGGER
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("[INFO]: Rank: %i -> PGSD: pgsd_get_nframes\n", rank);
#endif
    if (handle == NULL)
        {
        return 0;
        }
    if ( !frame_is_same(handle->cur_frame) )
        fprintf(stderr, "Frame numbers are not the same!\n");

    return handle->cur_frame;
    }


uint64_t pgsd_get_nnames(struct pgsd_handle* handle)
    {
#if PGSD_ACTIVATE_LOGGER
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("[INFO]: Rank: %i -> PGSD: pgsd_get_nnames\n", rank);
#endif
    if (handle == NULL)
        {
        return 0;
        }

    return handle->file_names.n_names;
    }


const struct pgsd_index_entry*
pgsd_find_chunk(struct pgsd_handle* handle, uint64_t frame, const char* name)
    {
#if PGSD_ACTIVATE_LOGGER
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("[INFO]: Rank: %i -> PGSD: pgsd_find_chunk\n", rank);
#endif
    if (handle == NULL)
        {
        return NULL;
        }
    if (name == NULL)
        {
        return NULL;
        }
    if (frame >= pgsd_get_nframes(handle))
        {
        return NULL;
        }
    if (handle->open_flags != PGSD_OPEN_READONLY)
        {
        int retval = pgsd_flush(handle);
        if (retval != PGSD_SUCCESS)
            {
            return NULL;
            }
        }

    uint16_t match_id = 0;
    if ( is_root() )
        {
        // find the id for the given name
        match_id = pgsd_name_id_map_find(&handle->name_map, name);
        }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&match_id, 1, MPI_UINT16_T, 0, MPI_COMM_WORLD);

    if (match_id == UINT16_MAX)
        {
        return NULL;
        }

    if (handle->header.pgsd_version >= pgsd_make_version(2, 0))
        {
        // pgsd 2.0 files sort the entire index
        // binary search for the index entry
        ssize_t L = 0;
        ssize_t R = handle->file_index.size - 1;
        struct pgsd_index_entry T;
        T.frame = frame;
        T.id = match_id;
        size_t m = 0;
        int found = 0;

        if ( is_root() )
            {
            while (L <= R)
                {
                m = (L + R) / 2;
                int cmp = pgsd_cmp_index_entry(handle->file_index.data + m, &T);
                if (cmp == -1)
                    {
                    L = m + 1;
                    }
                else if (cmp == 1)
                    {
                    R = m - 1;
                    }
                else
                    {
                    found = 1;
                    break;
                    }
                }
            }
        MPI_Bcast(&found, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!found)
            {
            return NULL;
            }
        MPI_Bcast(&m, 1, my_MPI_SIZE_T, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        return &(handle->file_index.data[m]);
        }
    else
        {
        // pgsd 1.0 file: use binary search to find the frame and linear search to find the entry
        size_t L = 0;
        size_t R = handle->file_index.size;

        int64_t cur_index = 0;
        if ( is_root() )
            {
            // progressively narrow the search window by halves
            do
                {
                size_t m = (L + R) / 2;

                if (frame < handle->file_index.data[m].frame)
                    {
                    R = m;
                    }
                else
                    {
                    L = m;
                    }
                } while ((R - L) > 1);

            // this finds L = the rightmost index with the desired frame
            // search all index entries with the matching frame
            for (cur_index = L; (cur_index >= 0) && (handle->file_index.data[cur_index].frame == frame);
                 cur_index--)
                {
                // if the frame matches, check the id
                if (match_id == handle->file_index.data[cur_index].id)
                    {
                    break;
                    }
                }

            // verify match was actually found
            if (cur_index < 0
                || handle->file_index.data[cur_index].frame != frame
                || handle->file_index.data[cur_index].id != match_id)
                {
                cur_index = -1;
                }
            }
        MPI_Bcast(&cur_index, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
        if (cur_index < 0)
            {
            return NULL;
            }
        return &(handle->file_index.data[cur_index]);
        }

    // if we got here, we didn't find the specified chunk
    return NULL;
    }

int pgsd_read_chunk(struct pgsd_handle* handle, 
                    void* data, 
                    const struct pgsd_index_entry* chunk, 
                    uint64_t N,
                    uint32_t M,
                    uint32_t offset, 
                    bool all)
    {
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);

#if PGSD_ACTIVATE_LOGGER
    printf("[INFO]: Rank: %i -> PGSD: pgsd_read_chunk\n", rank);
#endif
    if (handle == NULL)
        {
#if PGSD_ACTIVATE_LOGGER
        printf("[INFO]: Rank: %i -> PGSD: pgsd_read_chunk ERRORHANDLER invalid argument handle.\n", rank);
#endif
        return PGSD_ERROR_INVALID_ARGUMENT;
        }
    if (data == NULL)
        {
#if PGSD_ACTIVATE_LOGGER
        printf("[INFO]: Rank: %i -> PGSD: pgsd_read_chunk ERRORHANDLER invalid argument data.\n", rank);
#endif
        return PGSD_ERROR_INVALID_ARGUMENT;
        }
    if ( is_root() && chunk == NULL)
        {
#if PGSD_ACTIVATE_LOGGER
        printf("[INFO]: Rank: %i -> PGSD: pgsd_read_chunk ERRORHANDLER invalid argument chunck.\n", rank);
#endif
        return PGSD_ERROR_INVALID_ARGUMENT;
        }
    if (handle->open_flags != PGSD_OPEN_READONLY)
        {
        int retval = pgsd_flush(handle);
        if (retval != PGSD_SUCCESS)
            {
#if PGSD_ACTIVATE_LOGGER
        printf("[INFO]: Rank: %i -> PGSD: pgsd_read_chunk ERRORHANDLER flushing not possible.\n", rank);
#endif
            return retval;
            }
        }

    size_t N_global, M_global;
    uint8_t m_type;
    uint64_t stride = 0;
    int64_t m_location;

    if ( is_root() ){
        N_global = chunk->N;
        M_global = chunk->M;
        m_type = chunk->type;
        m_location = chunk->location;
    }

    MPI_Bcast(&N_global, 1, my_MPI_SIZE_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&M_global, 1, my_MPI_SIZE_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m_type, 1, MPI_UINT8_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m_location, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
    size_t size;

    // Offset needs multiplication with number of rows
    offset = offset * M;
    if ( all == false )
        {
        stride = 0;
        size = N_global * M_global * pgsd_sizeof_type((enum pgsd_type)m_type);
        }
    else if ( all == true )
        {   
        size = N * M * pgsd_sizeof_type((enum pgsd_type)m_type);
        stride = offset * pgsd_sizeof_type((enum pgsd_type)m_type);
        }

    if ( size == 0 )
        {
#if PGSD_ACTIVATE_LOGGER
        printf("[INFO]: Rank: %i -> PGSD: pgsd_read_chunk ERRORHANDLER corrupt file size.\n", rank);
#endif
        return PGSD_ERROR_FILE_CORRUPT;
        }
    if ( m_location == 0 )
        {
#if PGSD_ACTIVATE_LOGGER
        printf("[INFO]: Rank: %i -> PGSD: pgsd_read_chunk ERRORHANDLER corrupt file location.\n", rank);
#endif
        return PGSD_ERROR_FILE_CORRUPT;
        }

    // validate that we don't read past the end of the file
    if ((m_location + size + stride) > (uint64_t)handle->file_size)
        {
#if PGSD_ACTIVATE_LOGGER
        printf("[INFO]: Rank: %i -> PGSD: pgsd_read_chunk ERRORHANDLER corrupt file max size.\n", rank);
#endif
        return PGSD_ERROR_FILE_CORRUPT;
        }

    MPI_File_read_at(handle->fh, m_location + stride, data, size, MPI_BYTE, MPI_STATUS_IGNORE);

    return PGSD_SUCCESS;
    }

size_t pgsd_sizeof_type(enum pgsd_type type)
    {
    size_t val = 0;
    if (type == PGSD_TYPE_UINT8)
        {
        val = sizeof(uint8_t);
        }
    else if (type == PGSD_TYPE_UINT16)
        {
        val = sizeof(uint16_t);
        }
    else if (type == PGSD_TYPE_UINT32)
        {
        val = sizeof(uint32_t);
        }
    else if (type == PGSD_TYPE_UINT64)
        {
        val = sizeof(uint64_t);
        }
    else if (type == PGSD_TYPE_INT8)
        {
        val = sizeof(int8_t);
        }
    else if (type == PGSD_TYPE_INT16)
        {
        val = sizeof(int16_t);
        }
    else if (type == PGSD_TYPE_INT32)
        {
        val = sizeof(int32_t);
        }
    else if (type == PGSD_TYPE_INT64)
        {
        val = sizeof(int64_t);
        }
    else if (type == PGSD_TYPE_FLOAT)
        {
        val = sizeof(float);
        }
    else if (type == PGSD_TYPE_DOUBLE)
        {
        val = sizeof(double);
        }
    else
        {
        return 0;
        }
    return val;
    }

const char*
pgsd_find_matching_chunk_name(struct pgsd_handle* handle, const char* match, const char* prev)
    {
#if PGSD_ACTIVATE_LOGGER
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("[INFO]: Rank: %i -> PGSD: pgsd_find_matching_chunk_name\n", rank);
#endif
    if (handle == NULL)
        {
        return NULL;
        }
    if (match == NULL)
        {
        return NULL;
        }
    if (handle->file_names.n_names == 0)
        {
        return NULL;
        }
    if (handle->open_flags != PGSD_OPEN_READONLY)
        {
        int retval = pgsd_flush(handle);
        if (retval != PGSD_SUCCESS)
            {
            return NULL;
            }
        }

    // return nothing found if the name buffer is corrupt
    if (handle->file_names.data.data[handle->file_names.data.reserved - 1] != 0)
        {
        return NULL;
        }

    // determine search start index
    const char* search_str;
    if (prev == NULL)
        {
        search_str = handle->file_names.data.data;
        }
    else
        {
        // return not found if prev is not in range
        if (prev < handle->file_names.data.data)
            {
            return NULL;
            }
        if (prev >= (handle->file_names.data.data + handle->file_names.data.reserved))
            {
            return NULL;
            }

        if (handle->header.pgsd_version < pgsd_make_version(2, 0))
            {
            search_str = prev + PGSD_NAME_SIZE;
            }
        else
            {
            search_str = prev + strlen(prev) + 1;
            }
        }

    size_t match_len = strlen(match);

    while (search_str < (handle->file_names.data.data + handle->file_names.data.reserved))
        {
        if (search_str[0] != 0 && 0 == strncmp(match, search_str, match_len))
            {
            return search_str;
            }

        if (handle->header.pgsd_version < pgsd_make_version(2, 0))
            {
            search_str += PGSD_NAME_SIZE;
            }
        else
            {
            search_str += strlen(search_str) + 1;
            }
        }

    // searched past the end of the list, return NULL
    return NULL;
    }

uint64_t pgsd_get_maximum_write_buffer_size(struct pgsd_handle* handle)
    {
    if (handle == NULL)
        {
        return 0;
        }
    return handle->maximum_write_buffer_size;
    }

int pgsd_set_maximum_write_buffer_size(struct pgsd_handle* handle, uint64_t size)
    {
    if (handle == NULL || size == 0)
        {
        return PGSD_ERROR_INVALID_ARGUMENT;
        }

    handle->maximum_write_buffer_size = size;

    return PGSD_SUCCESS;
    }

uint64_t pgsd_get_index_entries_to_buffer(struct pgsd_handle* handle)
    {
    if (handle == NULL)
        {
        return 0;
        }
    return handle->index_entries_to_buffer;
    }

int pgsd_set_index_entries_to_buffer(struct pgsd_handle* handle, uint64_t number)
    {
    if (handle == NULL || number == 0)
        {
        return PGSD_ERROR_INVALID_ARGUMENT;
        }

    handle->index_entries_to_buffer = number;

    return PGSD_SUCCESS;
    }

// #endif // End ENABLE_MPI
