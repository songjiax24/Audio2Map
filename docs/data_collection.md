# Sayobot API — Data Collection

Reference for the Audio2Map dataset collector. Use a descriptive `User-Agent` and `Referer`; bulk downloads without them may be blocked.

## List API (`beatmaplist`)

- **GET** `https://api.sayobot.cn/beatmaplist`
- **POST** `https://api.sayobot.cn/?post` with `cmd=beatmaplist`

### Parameters (GET keys)

| Param | Key | Default | Description |
|-------|-----|---------|-------------|
| limit | L | 25 | Page size |
| offset | O | 0 | Start index; use `endid` from previous response |
| type | T | 1 | 1=hot, **2=new**, 3=packs, 4=search |
| class | C | all | **1=Ranked & Approved**, 2=Qualified, 4=Loved, … |
| mode | M | all | 1=std, 2=taiko, 4=ctb, **8=mania** |

Our collector uses `T=2` (new) + `C=1` (Ranked & Approved), then filters `modes & 8` before download.

### Response fields

| Field | Meaning |
|-------|---------|
| `data[]` | Beatmap set entries |
| `data[].sid` | Set ID |
| `data[].modes` | Bitmask of game modes |
| `data[].approved` | Rank status (1=ranked, 2=approved, …) |
| `endid` | Next offset; **0 = exhausted** |

Full parameter list: see archived `collect_id_setup.md` in git history or Sayobot docs.

## Download API

> Do not multi-thread downloads for a single map. After a cache miss, wait ~`filesize / 5` seconds before the next request.

| Variant | URL |
|---------|-----|
| MINI (used) | `https://dl.sayobot.cn/beatmaps/download/mini/{sid}` |
| No video | `https://dl.sayobot.cn/beatmaps/download/novideo/{sid}` |
| Full | `https://dl.sayobot.cn/beatmaps/download/full/{sid}` |

Preview audio: `https://a.sayobot.cn/preview/{sid}.mp3`

## Post-processing rules

Each kept set is stored as `raw/{sid}/`:

- Exactly **one `.mp3`**
- **One or more `.osu`** files, mania only (`Mode: 3`, `CircleSize: 4`)
- No subdirectories or extra files

Sets that fail normalization are deleted.

## Resume state

Progress is saved to `$AUDIO2MAP_DATA_ROOT/.collector/state.json`:

```json
{
  "offset": 0,
  "completed": [],
  "skipped_no_4k": [],
  "failed": {}
}
```

Re-run the same collect command to continue from `offset`.

## License note

Beatmap assets belong to their creators and osu! rights holders. Use only in compliance with osu! and Sayobot terms.
